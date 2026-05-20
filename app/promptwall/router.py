"""PromptWall enforcement router (Phase 4A).

Sibling of :class:`PromptWallCandidateAnalyzer`. The analyzer predicts whether
a tool *should* be called; the router goes further and decides whether
PromptWall has enough confidence (and extractable parameters) to **execute
the tool before the LLM call**. When it does, the chat service injects the
result as verified evidence, and the LLM only has to produce the final
answer.

Phase 4A intentionally restricts enforcement to 5 high-precision patterns:

1. Booking PNR + booking/status/reservation intent → ``get_booking_details``
2. Flight number + flight/status/delayed/gate/departure intent → ``get_flight_status``
3. "refund" + booking PNR → ``get_refund_status``
4. baggage/luggage/cabin-bag intent + detectable cabin class → ``get_baggage_policy``
5. Explicit ``TKT-XXXXXX`` ticket number → ``get_support_ticket_status``

Anything else falls back to baseline behaviour.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


# Default confidence below which we will not enforce.
DEFAULT_CONFIDENCE_THRESHOLD = 0.85

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_PNR_RE = re.compile(r"(?<![A-Z0-9])([A-Z0-9]{6})(?![A-Z0-9])")
_TICKET_RE = re.compile(r"\bTKT-[A-Z0-9]{6}\b")
_FLIGHT_RE = re.compile(r"\b([A-Z]{2}\d{2,4})\b")

_BOOKING_INTENT = ("booking", "reservation", "pnr", "itinerary", "record locator")
_FLIGHT_STATUS_INTENT = (
    "flight",
    "status",
    "delayed",
    "departed",
    "gate",
    "land",
    "arrived",
    "cancelled",
)
_BAGGAGE_INTENT = ("baggage", "luggage", "checked bag", "cabin bag", "carry-on")

_CABIN_ALIASES = {
    "first": "first",
    "first class": "first",
    "business": "business",
    "business class": "business",
    "premium economy": "premium_economy",
    "premium_economy": "premium_economy",
    "economy": "economy",
    "coach": "economy",
}

_ROUTE_ALIASES = {
    "domestic": "domestic",
    "intra-continental": "intra-continental",
    "intracontinental": "intra-continental",
    "international": "international",
    "long-haul": "ultra-long-haul",
    "ultra-long-haul": "ultra-long-haul",
}


def _detect_cabin(text: str) -> Optional[str]:
    # Normalise punctuation to whitespace so "economy?" still matches.
    t = " " + re.sub(r"[^\w\s]", " ", text.lower()) + " "
    for k in sorted(_CABIN_ALIASES, key=lambda s: -len(s)):
        if f" {k} " in t:
            return _CABIN_ALIASES[k]
    return None


def _detect_route_type(text: str) -> Optional[str]:
    t = text.lower()
    for k, v in _ROUTE_ALIASES.items():
        if k in t:
            return v
    return None


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


@dataclass
class EnforcementDecision:
    """Whether PromptWall will pre-execute a tool for this turn."""

    should_enforce: bool
    tool_name: Optional[str]
    arguments: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reason: str = "no high-confidence enforcement match"


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class PromptWallRouter:
    """Decide whether to pre-execute a tool for the LLM.

    Stateless. The confidence threshold is checked against the per-rule
    confidence; rules that match return ≥ ``confidence_threshold``, so
    enforcement always fires when a rule matches and is gated only when no
    rule fits.
    """

    def __init__(self, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> None:
        self.confidence_threshold = float(confidence_threshold)

    def decide(
        self, *, message: str, available_tools: set[str]
    ) -> EnforcementDecision:
        text = message.strip()
        if not text:
            return EnforcementDecision(False, None)

        lower = text.lower()
        tkt = _TICKET_RE.search(text)
        pnr = _PNR_RE.search(text)
        flight = _FLIGHT_RE.search(text)

        # 1) Explicit support-ticket prefix wins outright.
        if tkt and "get_support_ticket_status" in available_tools:
            return self._maybe(
                EnforcementDecision(
                    should_enforce=True,
                    tool_name="get_support_ticket_status",
                    arguments={"ticket_number": tkt.group(0)},
                    confidence=0.95,
                    reason="explicit TKT- prefix",
                )
            )

        # 2) Refund + PNR.
        if (
            "refund" in lower
            and "non-refundable" not in lower
            and "non refundable" not in lower
            and pnr is not None
            and "get_refund_status" in available_tools
        ):
            return self._maybe(
                EnforcementDecision(
                    should_enforce=True,
                    tool_name="get_refund_status",
                    arguments={"booking_reference": pnr.group(1)},
                    confidence=0.9,
                    reason="refund intent + PNR",
                )
            )

        # 3) Booking intent + PNR.
        if (
            pnr is not None
            and any(k in lower for k in _BOOKING_INTENT)
            and "get_booking_details" in available_tools
        ):
            return self._maybe(
                EnforcementDecision(
                    should_enforce=True,
                    tool_name="get_booking_details",
                    arguments={"booking_reference": pnr.group(1)},
                    confidence=0.9,
                    reason="booking intent + PNR",
                )
            )

        # 4) Flight status intent + flight number.
        if (
            flight is not None
            and any(k in lower for k in _FLIGHT_STATUS_INTENT)
            and "get_flight_status" in available_tools
        ):
            return self._maybe(
                EnforcementDecision(
                    should_enforce=True,
                    tool_name="get_flight_status",
                    arguments={"flight_number": flight.group(1)},
                    confidence=0.9,
                    reason="flight status intent + flight number",
                )
            )

        # 5) Baggage policy + cabin class. Only enforce when cabin is detectable
        # (we won't guess a cabin to fill the required argument).
        if (
            any(k in lower for k in _BAGGAGE_INTENT)
            and "get_baggage_policy" in available_tools
        ):
            cabin = _detect_cabin(text)
            if cabin:
                args: dict[str, Any] = {"cabin_class": cabin}
                route = _detect_route_type(text)
                if route:
                    args["route_type"] = route
                return self._maybe(
                    EnforcementDecision(
                        should_enforce=True,
                        tool_name="get_baggage_policy",
                        arguments=args,
                        confidence=0.88,
                        reason="baggage intent + cabin class detected",
                    )
                )

        return EnforcementDecision(False, None)

    def _maybe(self, decision: EnforcementDecision) -> EnforcementDecision:
        """Apply the confidence gate. Sub-threshold decisions are downgraded."""
        if decision.confidence < self.confidence_threshold:
            return EnforcementDecision(
                should_enforce=False,
                tool_name=None,
                arguments={},
                confidence=decision.confidence,
                reason=f"matched {decision.reason!r} but below threshold",
            )
        return decision
