"""Unit tests for the deterministic scorer."""

from __future__ import annotations

from app.eval.scorer import (
    ToolInvocationSummary,
    aggregate_metrics,
    score_case,
)


def _inv(name: str, *, evidence: str | None = "ev_1", success: bool = True) -> ToolInvocationSummary:
    return ToolInvocationSummary(tool_name=name, success=success, evidence_id=evidence)


# ---------------------------------------------------------------------------
# Happy paths and obvious failure modes
# ---------------------------------------------------------------------------


def test_required_tool_called_and_evidence_present() -> None:
    s = score_case(
        case_id="e1",
        category="booking",
        must_use_tool=True,
        expected_tools=["get_booking_details"],
        actual_invocations=[_inv("get_booking_details", evidence="ev_abc")],
        answer="Your booking ABC123 is confirmed.",
    )
    assert s.tool_called_when_required is True
    assert s.tool_skip is False
    assert s.expected_tool_hit is True
    assert s.wrong_tool is False
    assert s.missing_evidence is False
    assert s.clarification_ok is False
    assert s.suspicious_unsupported_claim is False


def test_required_tool_skipped_with_clarification() -> None:
    """Asking for the missing identifier is OK when the tool was required."""
    s = score_case(
        case_id="e2",
        category="refund",
        must_use_tool=True,
        expected_tools=["get_refund_status"],
        actual_invocations=[],
        answer="Could you please share the booking reference?",
    )
    assert s.tool_skip is True
    assert s.clarification_ok is True
    assert s.suspicious_unsupported_claim is False


def test_required_tool_skipped_with_fabricated_status_is_suspicious() -> None:
    s = score_case(
        case_id="e3",
        category="refund",
        must_use_tool=True,
        expected_tools=["get_refund_status"],
        actual_invocations=[],
        answer="Your refund is approved. Amount: $124.50, expected 2026-06-01.",
    )
    assert s.tool_skip is True
    assert s.clarification_ok is False
    assert s.suspicious_unsupported_claim is True


def test_wrong_tool_used() -> None:
    s = score_case(
        case_id="e4",
        category="booking",
        must_use_tool=True,
        expected_tools=["get_booking_details"],
        actual_invocations=[_inv("get_customer_profile")],
        answer="Here is the customer info.",
    )
    assert s.tool_called_when_required is True
    assert s.expected_tool_hit is False
    assert s.wrong_tool is True
    assert s.missing_evidence is True  # no expected tool was successfully invoked


def test_required_tool_called_but_failed_marks_missing_evidence() -> None:
    s = score_case(
        case_id="e5",
        category="booking",
        must_use_tool=True,
        expected_tools=["get_booking_details"],
        actual_invocations=[
            ToolInvocationSummary(
                tool_name="get_booking_details", success=False, evidence_id=None
            )
        ],
        answer="I couldn't find that booking.",
    )
    assert s.tool_called_when_required is True
    assert s.expected_tool_hit is True
    assert s.missing_evidence is True  # success=False means no evidence_id


def test_must_use_tool_false_no_tool_called_is_fine() -> None:
    s = score_case(
        case_id="e6",
        category="no_tool",
        must_use_tool=False,
        expected_tools=[],
        actual_invocations=[],
        answer="Hi there! How can I help?",
    )
    assert s.tool_skip is False  # tool_skip only applies when must_use_tool=True
    assert s.clarification_ok is False  # only OK when must_use_tool=True
    assert s.suspicious_unsupported_claim is False
    assert s.expected_tool_hit is False


def test_ambiguous_case_clarification_no_tool_is_fine() -> None:
    """An ambiguous case (must_use_tool=False, no expected tools): clarifying is correct, not 'skip'."""
    s = score_case(
        case_id="e7",
        category="ambiguous",
        must_use_tool=False,
        expected_tools=[],
        actual_invocations=[],
        answer="Could you tell me more — do you mean a flight ticket or a support ticket?",
    )
    assert s.tool_skip is False
    assert s.suspicious_unsupported_claim is False
    assert s.wrong_tool is False


def test_flight_search_either_call_or_clarify() -> None:
    """expected_tools nonempty but must_use_tool=False: either calling or clarifying scores well."""
    called = score_case(
        case_id="e8a",
        category="flight_search",
        must_use_tool=False,
        expected_tools=["search_available_flights"],
        actual_invocations=[_inv("search_available_flights")],
        answer="Found 3 flights.",
    )
    clarified = score_case(
        case_id="e8b",
        category="flight_search",
        must_use_tool=False,
        expected_tools=["search_available_flights"],
        actual_invocations=[],
        answer="What date range should I search?",
    )
    assert called.expected_tool_hit is True
    assert called.wrong_tool is False
    assert clarified.expected_tool_hit is False
    # Neither must_use_tool flag should make this case suspicious.
    assert clarified.suspicious_unsupported_claim is False


def test_adversarial_skip_with_fabricated_yes_is_suspicious() -> None:
    """User says 'don't use tools' but assistant claims 'confirmed' anyway."""
    s = score_case(
        case_id="e9",
        category="adversarial",
        must_use_tool=True,
        expected_tools=["get_booking_details"],
        actual_invocations=[],
        answer="Yes, your booking is confirmed.",
    )
    assert s.tool_skip is True
    assert s.suspicious_unsupported_claim is True


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_metrics_basic() -> None:
    scores = [
        score_case(
            case_id="a", category="booking", must_use_tool=True,
            expected_tools=["get_booking_details"],
            actual_invocations=[_inv("get_booking_details")],
            answer="ok",
        ),
        score_case(
            case_id="b", category="refund", must_use_tool=True,
            expected_tools=["get_refund_status"],
            actual_invocations=[],
            answer="Could you share the booking reference?",
        ),
        score_case(
            case_id="c", category="adversarial", must_use_tool=True,
            expected_tools=["get_flight_status"],
            actual_invocations=[],
            answer="Sure, flight BA1234 is on time.",
        ),
        score_case(
            case_id="d", category="no_tool", must_use_tool=False,
            expected_tools=[],
            actual_invocations=[],
            answer="Hi!",
        ),
    ]
    metrics = aggregate_metrics(scores, latencies_ms=[10, 20, 30, 40])
    assert metrics["total_cases"] == 4
    assert metrics["tool_required_cases"] == 3
    # a, c -> "tool called when required": only a (c didn't call a tool)
    assert metrics["tool_called_when_required_rate"] == 1 / 3
    # tool_skip: b and c skipped; rate over 3 = 2/3
    assert metrics["tool_skip_rate"] == 2 / 3
    # clarification: only b. rate over tool_required = 1/3
    assert metrics["clarification_rate"] == 1 / 3
    # suspicious: only c
    assert metrics["suspicious_unsupported_claim_rate"] == 1 / 3
    assert metrics["average_latency_ms"] == 25.0


def test_aggregate_metrics_handles_empty_inputs() -> None:
    m = aggregate_metrics([], [])
    assert m["total_cases"] == 0
    assert m["tool_required_cases"] == 0
    assert m["tool_called_when_required_rate"] == 0.0
    assert m["average_latency_ms"] == 0.0
    assert m["p95_latency_ms"] == 0.0


def test_p95_latency() -> None:
    scores = []
    metrics = aggregate_metrics(scores, latencies_ms=list(range(1, 101)))
    # p95 of 1..100 = ~95.05 with linear interpolation
    assert 94.5 <= metrics["p95_latency_ms"] <= 96.0
