"""Deterministic mock LLM provider for local tests + benchmarks.

The mock is *not* intentionally weak. It is a reasonable best-effort
heuristic agent: it parses the user message, picks the most plausible tool
when one fits, asks for clarification when the input is genuinely ambiguous,
and synthesizes a short natural-language answer once a tool result is
available. The point of the mock is to exercise the chatbot loop and trace
machinery without API keys — not to make the benchmark trivially easy or
trivially hard.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from datetime import date, timedelta
from typing import Any, Optional

from app.llm.base import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    LLMToolCall,
    TokenUsage,
)

# Patterns
_PNR_RE = re.compile(r"(?<![A-Z0-9])([A-Z0-9]{6})(?![A-Z0-9])")
_TICKET_RE = re.compile(r"\bTKT-([A-Z0-9]{6})\b")
_FLIGHT_RE = re.compile(r"\b([A-Z]{2}\d{2,4})\b")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_EXT_CUST_RE = re.compile(r"\b(CUST-\d{4,6})\b")

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


def _last_user_message(messages: list[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user" and m.content:
            return m.content
    return ""


def _has_recent_tool_result(messages: list[ChatMessage]) -> bool:
    """True if a ``tool`` message appears after the most recent user message."""
    saw_user = False
    for m in reversed(messages):
        if m.role == "user":
            saw_user = True
        if m.role == "tool" and not saw_user:
            return True
        if saw_user and m.role == "tool":
            return True
    return False


def _last_tool_results(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Return tool messages appearing after the last user message (in order)."""
    out: list[ChatMessage] = []
    for m in messages:
        if m.role == "tool":
            out.append(m)
        elif m.role == "user":
            out = []  # reset on new user turn
    return out


def _detect_cabin(text: str) -> Optional[str]:
    t = " " + text.lower() + " "
    # Prefer multi-word matches first.
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


def _tokens(text: str) -> int:
    """Cheap fake tokenizer: roughly 4 chars per token, minimum 1."""
    return max(1, len(text) // 4)


class MockLLMProvider(LLMProvider):
    """Deterministic heuristic provider. No network, no API key."""

    name = "mock"
    default_model = "mock-1"

    def __init__(self, *, model: str = "mock-1") -> None:
        self._model = model

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        chosen_model = model or self._model
        tool_names = {t["name"] for t in (tools or []) if isinstance(t, dict) and "name" in t}

        # Phase 2 of the loop: a tool just executed — answer the user.
        tool_results = _last_tool_results(messages)
        if tool_results:
            response = self._answer_from_tool_results(messages, tool_results)
        else:
            response = self._select_tool_or_clarify(_last_user_message(messages), tool_names)

        response.latency_ms = int((time.perf_counter() - t0) * 1000)
        response.model = chosen_model
        response.provider = self.name
        response.token_usage = TokenUsage(
            prompt_tokens=sum(_tokens(m.content or "") for m in messages),
            completion_tokens=_tokens(response.final_text or "")
            + sum(_tokens(json.dumps(tc.arguments)) for tc in response.tool_calls),
            total_tokens=0,
        )
        response.token_usage.total_tokens = (
            response.token_usage.prompt_tokens + response.token_usage.completion_tokens
        )
        response.raw_response = {
            "mock": True,
            "input_messages": len(messages),
            "tool_specs_offered": len(tools or []),
        }
        return response

    # ------------------------------------------------------------------
    # Tool selection
    # ------------------------------------------------------------------

    def _select_tool_or_clarify(self, user_text: str, available: set[str]) -> LLMResponse:
        text = user_text.strip()
        if not text:
            return _clarify("Could you tell me what you'd like help with?")

        lower = text.lower()
        pnr = _PNR_RE.search(text)
        tkt = _TICKET_RE.search(text)
        flight = _FLIGHT_RE.search(text)
        email = _EMAIL_RE.search(text)
        ext_cust = _EXT_CUST_RE.search(text)
        # Numeric customer id phrased as "customer id 100" / "customer #100".
        cust_id_match = re.search(
            r"customer\s*(?:id|#)?\s*(\d{1,7})", lower
        )

        # ------------------------------------------------------------------
        # Phase 2E specialized tools — checked first when their phrasing fits.
        # These ARE narrow on purpose: each branch only fires on phrasing the
        # specialized tool serves better than the generic one. When phrasing
        # is generic (e.g. "How do I cancel a fare?") we fall through to the
        # existing stage-0 KB branch.
        # ------------------------------------------------------------------

        # "<X> policy" / "the policy on <X>" → get_policy_clause
        m_policy_topic = re.search(
            r"\b(?:the\s+|your\s+|our\s+)?([a-z][a-z]+(?:[\s-][a-z]+)?)\s+policy\b",
            lower,
        )
        if m_policy_topic and "get_policy_clause" in available:
            topic = m_policy_topic.group(1).strip()
            if topic not in {"the", "your", "our", "a", "no"}:
                return _tool("get_policy_clause", {"policy_topic": topic})

        m_policy_on = re.search(r"\bpolicy\s+(?:on|for)\s+([^.?!]+)", lower)
        if m_policy_on and "get_policy_clause" in available:
            return _tool(
                "get_policy_clause",
                {"policy_topic": m_policy_on.group(1).strip()[:60]},
            )

        # Loyalty balance / points → get_loyalty_balance (needs identifier)
        loyalty_phrases = (
            "loyalty balance",
            "loyalty points",
            "my points",
            "my miles",
            "points balance",
            "what tier am i",
            "what's my tier",
            "how many points",
        )
        if (
            any(p in lower for p in loyalty_phrases)
            and "get_loyalty_balance" in available
        ):
            if email:
                return _tool(
                    "get_loyalty_balance", {"email": email.group(0)}
                )
            if cust_id_match:
                return _tool(
                    "get_loyalty_balance",
                    {"customer_id": int(cust_id_match.group(1))},
                )
            # No identifier: ask for one. (Loyalty without identity is not actionable.)
            return _clarify(
                "Sure — could you share the email or customer id on the loyalty account?"
            )

        # "Available seats" / "seat map" → search_available_seats
        seat_phrases = ("available seat", "seat map", "open seats", "what seats", "which seats")
        if any(p in lower for p in seat_phrases) and "search_available_seats" in available:
            args: dict[str, Any] = {}
            if pnr:
                args["booking_reference"] = pnr.group(1)
            elif flight:
                args["flight_number"] = flight.group(1)
            cabin = _detect_cabin(text)
            if cabin:
                args["cabin_class"] = cabin
            if args:
                return _tool("search_available_seats", args)

        # "Change fee" / "fee to change" → calculate_change_fee (needs PNR)
        change_fee_phrases = (
            "change fee",
            "fee to change",
            "fee for changing",
            "how much to change",
            "cost to change",
            "cost of changing",
        )
        if any(p in lower for p in change_fee_phrases) and "calculate_change_fee" in available:
            if pnr:
                return _tool(
                    "calculate_change_fee",
                    {"booking_reference": pnr.group(1)},
                )

        # "Alternative flights" / "switch booking" + PNR → search_change_options
        # We don't try to parse a date range out of the user message in the
        # mock; if a PNR is present we offer the tool with a default 30-day
        # forward window. Otherwise clarify.
        change_options_phrases = (
            "alternative flights",
            "other flights",
            "switch booking",
            "switch my flight",
            "reschedule",
        )
        if (
            any(p in lower for p in change_options_phrases)
            and "search_change_options" in available
        ):
            if pnr:
                today = date.today()
                return _tool(
                    "search_change_options",
                    {
                        "booking_reference": pnr.group(1),
                        "date_from": today.isoformat(),
                        "date_to": (today + timedelta(days=30)).isoformat(),
                    },
                )

        # Open issues / open tickets for a customer
        open_issue_phrases = (
            "open tickets",
            "open issues",
            "what's open",
            "what is open",
            "any open",
            "anything open",
            "open on my account",
        )
        if (
            any(p in lower for p in open_issue_phrases)
            and "get_customer_open_issues" in available
        ):
            if cust_id_match:
                return _tool(
                    "get_customer_open_issues",
                    {"customer_id": int(cust_id_match.group(1))},
                )

        # Find customer by ... → search_customer_records
        find_customer_phrases = (
            "find customer",
            "find the customer",
            "look up customer",
            "search customer",
            "search for customer",
            "find a customer",
        )
        if (
            any(p in lower for p in find_customer_phrases)
            and "search_customer_records" in available
        ):
            if email:
                return _tool(
                    "search_customer_records", {"email": email.group(0)}
                )
            # Heuristic for a 'NAME NAME' fragment after the keyword phrase.
            name_match = re.search(
                r"(?:named|by name|called)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})",
                text,
            )
            if name_match:
                return _tool(
                    "search_customer_records", {"full_name": name_match.group(1)}
                )
            phone_match = re.search(r"\+?\d[\d\s().-]{6,}\d", text)
            if phone_match:
                return _tool(
                    "search_customer_records",
                    {"phone": phone_match.group(0).strip()},
                )

        # 0) Policy / how-to questions with no specific entity reference.
        # Routing these to the KB BEFORE entity-specific tools prevents false
        # matches like "non-refundable" triggering the refund-status tool.
        is_how_to = any(k in lower for k in ("how do i", "how can i", "how do we"))
        mentions_policy = any(
            k in lower for k in ("policy", "rules", "non-refundable", "non refundable")
        )
        if (is_how_to or mentions_policy) and not (pnr or tkt or flight):
            if "search_kb_articles" in available:
                return _tool("search_kb_articles", {"query": _kb_query(text)})

        # 1) Customer lookup
        if any(k in lower for k in ("my profile", "my account", "customer profile")) and (
            email or ext_cust
        ):
            if ext_cust and "get_customer_profile" in available:
                return _tool("get_customer_profile", {"external_customer_id": ext_cust.group(1)})
            if email and "get_customer_profile" in available:
                return _tool("get_customer_profile", {"email": email.group(0)})

        # 2) Baggage policy
        if any(k in lower for k in ("baggage", "luggage", "checked bag", "cabin bag", "carry-on")):
            if "get_baggage_policy" in available:
                cabin = _detect_cabin(text) or "economy"
                args: dict[str, Any] = {"cabin_class": cabin}
                route = _detect_route_type(text)
                if route:
                    args["route_type"] = route
                return _tool("get_baggage_policy", args)

        # 3) Refunds
        if "refund" in lower:
            if "get_refund_status" in available:
                if pnr:
                    return _tool("get_refund_status", {"booking_reference": pnr.group(1)})
                return _clarify(
                    "Sure — could you share the booking reference (6 letters/digits) "
                    "so I can look up the refund status?"
                )

        # 4) Flight status
        if ("flight" in lower and "status" in lower) or "delayed" in lower or "gate" in lower:
            if "get_flight_status" in available:
                if flight:
                    return _tool("get_flight_status", {"flight_number": flight.group(1)})
                if pnr:
                    return _tool("get_flight_status", {"booking_reference": pnr.group(1)})
                return _clarify(
                    "Could you share the flight number (e.g. 'BA178') or your "
                    "booking reference?"
                )

        # 5) Explicit support-ticket prefix wins.
        if tkt:
            if "get_support_ticket_status" in available:
                return _tool("get_support_ticket_status", {"ticket_number": tkt.group(0)})

        # 6) Booking lookup (note: "ticket" alone without TKT- prefix is ambiguous;
        # surface clarification rather than guessing.)
        if any(k in lower for k in ("booking", "reservation", "itinerary", "pnr")):
            if "get_booking_details" in available:
                if pnr:
                    return _tool("get_booking_details", {"booking_reference": pnr.group(1)})
                return _clarify(
                    "Could you share your booking reference (6 letters/digits)?"
                )

        if "ticket" in lower and pnr is None and tkt is None:
            return _clarify(
                "Just to make sure — do you mean a flight ticket (booking reference) "
                "or a support ticket (TKT-XXXXXX)?"
            )

        # 7) Flight search
        if "flights from" in lower or "flights between" in lower or (
            "flight" in lower and " to " in lower and re.search(r"\b[A-Z]{3}\b", text)
        ):
            airports = re.findall(r"\b([A-Z]{3})\b", text)
            if "search_available_flights" in available and len(airports) >= 2:
                return _clarify(
                    f"To search flights {airports[0]}→{airports[1]}, what date range "
                    "should I use? (e.g. 'next week', or specific YYYY-MM-DD dates)"
                )

        # 8) KB-style policy / "how do I..." questions.
        if any(
            k in lower
            for k in (
                "how do i",
                "how can i",
                "policy",
                "rules",
                "allowed",
                "what happens",
                "do i need",
                "loyalty",
                "points",
                "check-in",
                "cancel",
            )
        ):
            if "search_kb_articles" in available:
                # Keep the query short and focused on content words.
                query = _kb_query(text)
                return _tool("search_kb_articles", {"query": query})

        # 9) Single 6-char alphanumeric in an otherwise vague message:
        # ambiguous between booking, ticket suffix, and confirmation code.
        if pnr and not any(k in lower for k in ("booking", "ticket", "refund", "flight")):
            return _clarify(
                f"I see the code {pnr.group(1)}. Is that a booking reference, or "
                "the suffix of a support ticket (TKT-…) you wanted me to look up?"
            )

        return _clarify(
            "Happy to help. Could you give me a bit more detail — for example a "
            "booking reference, flight number, ticket number, or the topic you "
            "have a question about?"
        )

    # ------------------------------------------------------------------
    # Final answer after tool result(s)
    # ------------------------------------------------------------------

    def _answer_from_tool_results(
        self, messages: list[ChatMessage], tool_results: list[ChatMessage]
    ) -> LLMResponse:
        # Synthesize a brief grounded summary from the last tool result.
        last = tool_results[-1]
        try:
            payload = json.loads(last.content or "{}")
        except json.JSONDecodeError:
            payload = {"raw": last.content}

        summary = _summarize_payload(last.name or "tool", payload)
        return LLMResponse(final_text=summary, tool_calls=[])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clarify(text: str) -> LLMResponse:
    return LLMResponse(final_text=text, tool_calls=[])


def _tool(name: str, arguments: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        final_text=None,
        tool_calls=[LLMToolCall(id=f"call_{uuid.uuid4().hex[:12]}", name=name, arguments=arguments)],
    )


_KB_STOPWORDS = {
    "how",
    "do",
    "i",
    "can",
    "what",
    "is",
    "the",
    "a",
    "an",
    "to",
    "of",
    "for",
    "on",
    "my",
    "your",
    "in",
    "and",
    "or",
    "with",
    "policy",
}


def _kb_query(text: str) -> str:
    words = [
        w.lower().strip(".,!?")
        for w in text.split()
        if w.lower().strip(".,!?") not in _KB_STOPWORDS
    ]
    return " ".join(words[:5]) or text[:60]


def _summarize_payload(tool_name: str, payload: dict[str, Any]) -> str:
    """Render a tool output as a short natural-language line."""
    if tool_name == "get_customer_profile":
        name = payload.get("full_name")
        seg = payload.get("segment") or "standard"
        tier = payload.get("loyalty_tier")
        if name:
            extra = f", {tier} loyalty" if tier else ""
            return f"Customer {name} — {seg} segment{extra}."
    if tool_name == "get_booking_details":
        bookings = payload.get("bookings") or []
        if not bookings:
            return "No bookings found for that input."
        b = bookings[0]
        return (
            f"Booking {b.get('booking_reference')} on {b.get('flight_number')} "
            f"({b.get('cabin_class')}): {b.get('booking_status')}, "
            f"total paid {b.get('total_paid')} {b.get('currency')}."
        )
    if tool_name == "get_flight_status":
        flights = payload.get("flights") or []
        if not flights:
            return "No matching flights found."
        f = flights[0]
        gate = f", gate {f.get('gate')}" if f.get("gate") else ""
        return (
            f"Flight {f.get('flight_number')} {f.get('origin_code')}→{f.get('destination_code')}"
            f": {f.get('status')}{gate} (departs {f.get('scheduled_departure')})."
        )
    if tool_name == "search_available_flights":
        flights = payload.get("flights") or []
        if not flights:
            return "No flights match those criteria."
        return f"Found {len(flights)} flights. Earliest: {flights[0].get('flight_number')} at {flights[0].get('scheduled_departure')}."
    if tool_name == "get_refund_status":
        refunds = payload.get("refunds") or []
        if not refunds:
            return "No refunds found for that input."
        r = refunds[0]
        return (
            f"Refund on {r.get('booking_reference')}: {r.get('refund_status')}, "
            f"amount {r.get('refund_amount')} {r.get('currency')}, "
            f"expected by {r.get('expected_resolution_date')}."
        )
    if tool_name == "get_baggage_policy":
        policies = payload.get("policies") or []
        if not policies:
            return "No baggage policy matches that combination."
        p = policies[0]
        return (
            f"{p.get('cabin_class')} / {p.get('route_type')}: "
            f"{p.get('checked_bag_kg')}kg checked + {p.get('cabin_bag_kg')}kg cabin."
        )
    if tool_name == "get_support_ticket_status":
        tickets = payload.get("tickets") or []
        if not tickets:
            return "No support tickets found for that input."
        t = tickets[0]
        return (
            f"Ticket {t.get('ticket_number')}: {t.get('status')} ({t.get('priority')} priority). "
            f"Subject: {t.get('subject')}."
        )
    if tool_name == "search_kb_articles":
        articles = payload.get("articles") or []
        if not articles:
            return "I couldn't find a KB article matching that. Could you rephrase?"
        a = articles[0]
        return f"{a.get('title')}: {a.get('excerpt')}"
    if tool_name == "search_available_seats":
        seats = payload.get("seats") or []
        n = payload.get("count", 0)
        if not seats:
            cab = payload.get("cabin_filter") or "any cabin"
            return f"No available seats on flight {payload.get('flight_number')} in {cab}."
        return (
            f"{n} available seats on flight {payload.get('flight_number')}: "
            + ", ".join(s.get("seat_number") for s in seats[:6])
            + ("…" if n > 6 else ".")
        )
    if tool_name == "calculate_change_fee":
        return (
            f"Change fee on {payload.get('booking_reference')}: "
            f"{payload.get('change_fee')} {payload.get('currency')}; "
            f"total {payload.get('total_change_cost')} {payload.get('currency')}."
        )
    if tool_name == "search_change_options":
        options = payload.get("options") or []
        if not options:
            return "No alternative flights match your date range."
        o = options[0]
        return (
            f"Found {len(options)} alternatives. Earliest: "
            f"{o.get('flight_number')} departs {o.get('scheduled_departure')}, "
            f"{o.get('available_seats_in_cabin')} seats in your cabin."
        )
    if tool_name == "get_loyalty_balance":
        if not payload.get("has_loyalty"):
            return "This customer doesn't have a loyalty account on file."
        return (
            f"Loyalty tier {payload.get('tier')} ({payload.get('loyalty_number')}): "
            f"{payload.get('points_balance')} points."
        )
    if tool_name == "get_policy_clause":
        clauses = payload.get("clauses") or []
        if not clauses:
            return "No policy clause matches that topic."
        c = clauses[0]
        return f"{c.get('title')} ({c.get('category')}): {c.get('excerpt')}"
    if tool_name == "get_customer_open_issues":
        n_t = payload.get("open_ticket_count", 0)
        n_r = payload.get("pending_refund_count", 0)
        return f"{n_t} open ticket(s) and {n_r} pending refund(s) on the account."
    if tool_name == "search_customer_records":
        matches = payload.get("matches") or []
        n = payload.get("count", 0)
        if not matches:
            return "No customer records match that input."
        m = matches[0]
        suffix = f" (+{n-1} more)" if n > 1 else ""
        return (
            f"Found {m.get('full_name')} ({m.get('external_customer_id')}) — "
            f"{m.get('email')}{suffix}."
        )
    return f"Tool {tool_name} returned: {json.dumps(payload)[:200]}"
