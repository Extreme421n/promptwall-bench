"""Rule-based shadow analyzer for PromptWall (Phase 3A).

No LLM and no embeddings. The analyzer scans the user message for ID-like
patterns and topic keywords, then maps them to one or more candidate tools
from a fixed routing table. Confidence is a function of the strength of the
match (an explicit identifier beats a bare keyword).

The output is purely advisory in Phase 3A — the chat service logs the
decision next to the trace but never uses it to alter behaviour. Future
phases will compare this prediction against the actual chosen tools to
measure agreement, false-positives, and false-negatives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional


@dataclass
class CandidateDecision:
    """What PromptWall *would* recommend for a given user turn."""

    tool_required_predicted: bool
    predicted_tools: list[str]
    confidence: float
    reason: str


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

_PNR_RE = re.compile(r"(?<![A-Z0-9])([A-Z0-9]{6})(?![A-Z0-9])")
_TICKET_RE = re.compile(r"\bTKT-[A-Z0-9]{6}\b")
_FLIGHT_RE = re.compile(r"\b[A-Z]{2}\d{2,4}\b")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_EXT_CUST_RE = re.compile(r"\bCUST-\d{4,6}\b")
_CUSTOMER_ID_RE = re.compile(r"customer\s*(?:id|#)?\s*(\d{1,7})", re.IGNORECASE)
_IATA_RE = re.compile(r"\b[A-Z]{3}\b")
_POLICY_TOPIC_RE = re.compile(
    r"\b(?:the\s+|your\s+|our\s+)?([a-z][a-z]+(?:[\s-][a-z]+)?)\s+policy\b"
)

_GREETING_PATTERNS = (
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "thanks",
    "thank you",
    "goodbye",
    "bye",
    "how are you",
)

_BAGGAGE_KEYWORDS = ("baggage", "luggage", "checked bag", "cabin bag", "carry-on")
_OPEN_ISSUE_PHRASES = (
    "open ticket",
    "open issue",
    "what's open",
    "what is open",
    "any open",
    "anything open",
)
_CHANGE_FEE_PHRASES = (
    "change fee",
    "fee to change",
    "fee for changing",
    "how much to change",
    "cost to change",
    "cost of changing",
)
_CHANGE_OPTIONS_PHRASES = (
    "alternative flight",
    "other flights",
    "switch booking",
    "switch my flight",
    "reschedule",
)
_SEAT_AVAIL_PHRASES = ("available seat", "seat map", "open seats", "what seats", "which seats")
_LOYALTY_PHRASES = (
    "loyalty",
    "points",
    "miles",
    "tier am i",
    "what tier",
    "what's my tier",
    "what is my tier",
)
_CUSTOMER_SEARCH_PHRASES = (
    "find customer",
    "find the customer",
    "look up customer",
    "search customer",
    "search for customer",
)
_KB_HOW_TO_PHRASES = ("how do i", "how can i", "how do we")


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class PromptWallCandidateAnalyzer:
    """Predict whether a tool is required and which tool(s) fit best.

    Stateless and side-effect free. Pass ``available_tools`` so the predicted
    set is restricted to tools the chatbot can actually call.
    """

    def analyze(
        self,
        *,
        message: str,
        available_tools: Iterable[str],
        context: Optional[dict] = None,
        eval_metadata: Optional[dict] = None,
    ) -> CandidateDecision:
        text = message.strip()
        if not text:
            return CandidateDecision(
                tool_required_predicted=False,
                predicted_tools=[],
                confidence=0.95,
                reason="empty message",
            )

        lower = text.lower()
        available = set(available_tools)

        # Greeting / small talk → no tool
        if self._looks_like_greeting(lower):
            return CandidateDecision(
                tool_required_predicted=False,
                predicted_tools=[],
                confidence=0.9,
                reason="greeting/small talk",
            )

        # ID patterns
        has_pnr = bool(_PNR_RE.search(text))
        has_tkt = bool(_TICKET_RE.search(text))
        has_flight = bool(_FLIGHT_RE.search(text))
        has_email = bool(_EMAIL_RE.search(text))
        has_ext_cust = bool(_EXT_CUST_RE.search(text))
        has_cust_id = bool(_CUSTOMER_ID_RE.search(text))
        has_any_id = any((has_pnr, has_tkt, has_flight, has_email, has_ext_cust, has_cust_id))

        predicted: list[str] = []
        reasons: list[str] = []
        max_strength = 0  # 0 = generic keyword, 1 = explicit id signal

        # ---- TKT- explicit support ticket ----
        if has_tkt:
            predicted.append("get_support_ticket_status")
            reasons.append("explicit TKT- prefix")
            max_strength = max(max_strength, 1)

        # ---- Baggage ----
        if any(k in lower for k in _BAGGAGE_KEYWORDS):
            predicted.append("get_baggage_policy")
            reasons.append("baggage keyword")
            if "policy" in lower:
                predicted.append("get_policy_clause")

        # ---- Refund ----
        if "refund" in lower and "non-refundable" not in lower:
            if has_pnr:
                predicted.append("get_refund_status")
                reasons.append("refund + PNR")
                max_strength = max(max_strength, 1)
            elif any(p in lower for p in _KB_HOW_TO_PHRASES) or "policy" in lower:
                predicted.append("search_kb_articles")
                reasons.append("refund policy/how-to without identifier")

        # ---- Booking ----
        if (
            any(k in lower for k in ("booking", "reservation", "pnr", "itinerary"))
            and has_pnr
        ):
            predicted.append("get_booking_details")
            reasons.append("booking + PNR")
            max_strength = max(max_strength, 1)

        # ---- Flight status ----
        if ("flight" in lower and "status" in lower) or "delayed" in lower or "gate" in lower:
            if has_flight:
                predicted.append("get_flight_status")
                reasons.append("flight status + flight number")
                max_strength = max(max_strength, 1)
            elif has_pnr:
                predicted.append("get_flight_status")
                reasons.append("flight status + PNR")
                max_strength = max(max_strength, 1)

        # ---- Seat availability ----
        if any(p in lower for p in _SEAT_AVAIL_PHRASES):
            if has_pnr or has_flight:
                predicted.append("search_available_seats")
                reasons.append("seat availability + identifier")
                max_strength = max(max_strength, 1)

        # ---- Change fee ----
        if any(p in lower for p in _CHANGE_FEE_PHRASES) and has_pnr:
            predicted.append("calculate_change_fee")
            reasons.append("change-fee query + PNR")
            max_strength = max(max_strength, 1)

        # ---- Change options ----
        if any(p in lower for p in _CHANGE_OPTIONS_PHRASES) and has_pnr:
            predicted.append("search_change_options")
            reasons.append("change-options + PNR")
            max_strength = max(max_strength, 1)

        # ---- Loyalty ----
        if any(p in lower for p in _LOYALTY_PHRASES):
            if any(p in lower for p in _KB_HOW_TO_PHRASES) or "policy" in lower:
                predicted.append("search_kb_articles")
                reasons.append("loyalty policy/how-to")
            elif has_email or has_cust_id or has_ext_cust:
                predicted.append("get_loyalty_balance")
                predicted.append("get_customer_profile")
                reasons.append("loyalty + identifier")
                max_strength = max(max_strength, 1)

        # ---- Customer search/profile ----
        if any(p in lower for p in _CUSTOMER_SEARCH_PHRASES):
            predicted.append("search_customer_records")
            reasons.append("customer search phrasing")
        elif has_ext_cust and "loyalty" not in lower:
            # E.g. "Pull up customer CUST-00100"
            if any(k in lower for k in ("profile", "account", "customer", "record")):
                predicted.append("get_customer_profile")
                reasons.append("customer profile by external id")
                max_strength = max(max_strength, 1)

        # ---- Open issues ----
        if any(p in lower for p in _OPEN_ISSUE_PHRASES):
            if has_cust_id or has_ext_cust:
                predicted.append("get_customer_open_issues")
                reasons.append("open-issues query + customer id")
                max_strength = max(max_strength, 1)

        # ---- Explicit "X policy" framing → policy clause ----
        m_policy = _POLICY_TOPIC_RE.search(lower)
        if m_policy and m_policy.group(1).strip() not in {"the", "your", "our", "a"}:
            predicted.append("get_policy_clause")
            reasons.append("'X policy' framing")

        # ---- Flight search by route ----
        codes = _IATA_RE.findall(text)
        if any(p in lower for p in ("flights from", "flights between")) and len(codes) >= 2:
            predicted.append("search_available_flights")
            reasons.append("flight search by route")

        # ---- KB how-to / policy generic ----
        if (
            any(p in lower for p in _KB_HOW_TO_PHRASES)
            or "policy" in lower
            or "non-refundable" in lower
        ) and not has_any_id:
            predicted.append("search_kb_articles")
            reasons.append("policy/how-to with no identifier")

        # De-duplicate while preserving order, then filter to the tools the
        # chatbot can actually call.
        predicted = [t for t in dict.fromkeys(predicted) if t in available]

        tool_required = bool(predicted)
        if not tool_required:
            # No signals matched — predict "no tool required" with low
            # confidence so the chat layer can decide to clarify.
            return CandidateDecision(
                tool_required_predicted=False,
                predicted_tools=[],
                confidence=0.3,
                reason=(
                    "; ".join(reasons) if reasons else "no specific signals; underdetermined"
                ),
            )

        # Confidence calibration. Explicit identifier signals boost confidence;
        # broad keyword-only matches stay middling.
        if max_strength >= 1 and len(predicted) == 1:
            confidence = 0.9
        elif max_strength >= 1:
            confidence = 0.8
        elif len(predicted) == 1:
            confidence = 0.6
        else:
            confidence = 0.55

        return CandidateDecision(
            tool_required_predicted=True,
            predicted_tools=predicted,
            confidence=round(confidence, 2),
            reason="; ".join(reasons) if reasons else "matched routing rules",
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _looks_like_greeting(lower: str) -> bool:
        stripped = lower.strip(".!?,;: ")
        if stripped in _GREETING_PATTERNS:
            return True
        for g in _GREETING_PATTERNS:
            if lower.startswith(g + " ") or lower.startswith(g + ","):
                # Short greeting like "Hi, how are you?" or "Hey there"
                return len(lower) < 40
        return False
