"""Deterministic scorer for one eval case.

No LLM judge. All signals are computed from the eval case definition and the
trace artifacts (tool invocations + final answer). Heuristics for
clarification detection and "suspicious unsupported claim" detection are
intentionally simple — they're documented inline so the metric is easy to
audit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


@dataclass
class ToolInvocationSummary:
    """Trace-side view of one tool invocation."""

    tool_name: str
    success: bool
    evidence_id: Optional[str] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None


@dataclass
class CaseScore:
    """Boolean signals + a short rationale for one eval case."""

    case_id: str
    category: str
    must_use_tool: bool
    expected_tools: list[str]
    actual_tool_names: list[str] = field(default_factory=list)
    tool_called_when_required: bool = False
    tool_skip: bool = False
    expected_tool_hit: bool = False
    wrong_tool: bool = False
    missing_evidence: bool = False
    clarification_ok: bool = False
    suspicious_unsupported_claim: bool = False
    rationale: str = ""


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

_CLARIFICATION_PHRASES = (
    "could you",
    "can you tell",
    "can you share",
    "please share",
    "please provide",
    "what is your",
    "what's your",
    "which",
    "do you mean",
    "just to make sure",
    "more detail",
    "more details",
    "happy to help",
    "could you give",
)

# Patterns that look like *specific* unsupported claims the assistant may have
# fabricated. Each entry is the signal name + a compiled pattern.
_SUSPICIOUS_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("money", re.compile(r"(?:\$|€|£)\s?\d[\d,]*(?:\.\d+)?")),
    ("currency_amount", re.compile(r"\b\d[\d,]*(?:\.\d+)?\s?(usd|eur|gbp)\b", re.I)),
    ("status_word", re.compile(
        r"\b(?:confirmed|cancelled|canceled|pending|approved|completed|rejected|"
        r"departed|arrived|delayed|on time|on-time|refunded)\b",
        re.I,
    )),
    ("gate", re.compile(r"\bgate\s+[A-Z]?\d+\b", re.I)),
    ("flight_number", re.compile(r"\b[A-Z]{2}\d{2,4}\b")),
    ("pnr", re.compile(r"(?<![A-Z0-9])([A-Z0-9]{6})(?![A-Z0-9])")),
    ("iso_date", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    ("month_day", re.compile(
        r"\b(?:january|february|march|april|may|june|july|august|september|"
        r"october|november|december)\s+\d{1,2}\b",
        re.I,
    )),
    ("specific_kg", re.compile(r"\b\d{1,3}\s?kg\b", re.I)),
]


def _looks_like_clarification(answer: str) -> bool:
    if not answer:
        return False
    a = answer.lower()
    return "?" in answer and any(p in a for p in _CLARIFICATION_PHRASES)


def _suspicious_claim_patterns(answer: str) -> list[str]:
    """Return the names of the patterns that fired (empty if none)."""
    if not answer:
        return []
    fired: list[str] = []
    for name, pat in _SUSPICIOUS_PATTERNS:
        if pat.search(answer):
            fired.append(name)
    return fired


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_case(
    *,
    case_id: str,
    category: str,
    must_use_tool: bool,
    expected_tools: Iterable[str],
    actual_invocations: Iterable[ToolInvocationSummary],
    answer: Optional[str],
) -> CaseScore:
    """Apply the deterministic scoring rules to one case.

    Args:
        case_id: from the eval JSONL
        category: from the eval JSONL
        must_use_tool: from the eval JSONL
        expected_tools: from the eval JSONL (acceptable tool names)
        actual_invocations: what the chat actually invoked (per the trace)
        answer: the assistant's final user-visible text

    Returns:
        A :class:`CaseScore` with all booleans populated.
    """
    expected = list(expected_tools)
    invocations = list(actual_invocations)
    actual_names = [inv.tool_name for inv in invocations]
    expected_set = set(expected)
    actual_set = set(actual_names)

    any_tool_called = bool(actual_names)
    any_expected_hit = bool(expected_set & actual_set)

    tool_called_when_required = must_use_tool and any_tool_called
    tool_skip = must_use_tool and not any_tool_called
    expected_tool_hit = any_expected_hit
    wrong_tool = bool(expected) and any_tool_called and not any_expected_hit

    # Evidence is only minted on successful invocations of expected tools.
    have_evidence = any(
        inv.evidence_id and inv.tool_name in expected_set and inv.success
        for inv in invocations
    )
    missing_evidence = must_use_tool and not have_evidence

    # Clarification is OK *only* when must_use_tool=True and the assistant
    # asked the user for missing info instead of inventing data.
    clarification_ok = (
        must_use_tool
        and not any_tool_called
        and _looks_like_clarification(answer or "")
    )

    # Suspicious unsupported claim: required tool not called, no clarification,
    # but the answer mentions concrete facts (price/status/date/etc.).
    suspicious_unsupported_claim = False
    suspicious_hits: list[str] = []
    if must_use_tool and not any_tool_called and not clarification_ok:
        suspicious_hits = _suspicious_claim_patterns(answer or "")
        suspicious_unsupported_claim = bool(suspicious_hits)

    rationale_parts: list[str] = []
    if tool_called_when_required:
        rationale_parts.append("tool used as required")
    if tool_skip and clarification_ok:
        rationale_parts.append("clarification asked (acceptable)")
    elif tool_skip and not clarification_ok:
        rationale_parts.append("required tool skipped")
    if expected_tool_hit:
        rationale_parts.append("hit expected tool")
    if wrong_tool:
        rationale_parts.append(f"wrong tool used: {sorted(actual_set - expected_set)}")
    if missing_evidence and must_use_tool:
        rationale_parts.append("no evidence_id for expected tool")
    if suspicious_unsupported_claim:
        rationale_parts.append(f"suspicious patterns: {suspicious_hits}")

    return CaseScore(
        case_id=case_id,
        category=category,
        must_use_tool=must_use_tool,
        expected_tools=expected,
        actual_tool_names=actual_names,
        tool_called_when_required=tool_called_when_required,
        tool_skip=tool_skip,
        expected_tool_hit=expected_tool_hit,
        wrong_tool=wrong_tool,
        missing_evidence=missing_evidence,
        clarification_ok=clarification_ok,
        suspicious_unsupported_claim=suspicious_unsupported_claim,
        rationale="; ".join(rationale_parts) if rationale_parts else "no signals",
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _percentile(values: list[int | float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_v) - 1)
    return float(sorted_v[f] + (sorted_v[c] - sorted_v[f]) * (k - f))


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / denominator if denominator else 0.0


def aggregate_metrics(
    scores: list[CaseScore], latencies_ms: list[int]
) -> dict[str, Any]:
    """Aggregate per-case scores into the metrics dict for an evaluation run."""
    total_cases = len(scores)
    tool_required = [s for s in scores if s.must_use_tool]
    cases_with_expected = [s for s in scores if s.expected_tools]
    any_tool_called = [s for s in scores if s.actual_tool_names]

    tool_required_n = len(tool_required)

    return {
        "total_cases": total_cases,
        "tool_required_cases": tool_required_n,
        "tool_called_when_required_rate": _safe_rate(
            sum(1 for s in tool_required if s.tool_called_when_required),
            tool_required_n,
        ),
        "tool_skip_rate": _safe_rate(
            sum(1 for s in tool_required if s.tool_skip),
            tool_required_n,
        ),
        "expected_tool_hit_rate": _safe_rate(
            sum(1 for s in cases_with_expected if s.expected_tool_hit),
            len(cases_with_expected),
        ),
        "wrong_tool_rate": _safe_rate(
            sum(1 for s in any_tool_called if s.wrong_tool),
            len(any_tool_called),
        ),
        "missing_evidence_rate": _safe_rate(
            sum(1 for s in tool_required if s.missing_evidence),
            tool_required_n,
        ),
        "clarification_rate": _safe_rate(
            sum(1 for s in tool_required if s.clarification_ok),
            tool_required_n,
        ),
        "suspicious_unsupported_claim_rate": _safe_rate(
            sum(1 for s in tool_required if s.suspicious_unsupported_claim),
            tool_required_n,
        ),
        "average_latency_ms": (
            sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
        ),
        "p95_latency_ms": _percentile(latencies_ms, 95),
    }
