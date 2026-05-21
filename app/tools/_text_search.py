"""Phase 6C-1 — lightweight, deterministic text-retrieval helpers.

These helpers improve how the seven Phase 6B-4 text-retrieval tools handle
realistic user wording. They are intentionally NOT an LLM and do not depend on
any external service: just normalization, a hand-curated synonym map, and a
weighted-substring scoring pass.

Public surface:

* :func:`normalize`       — lowercase / strip-punctuation / collapse-whitespace
* :func:`expand_query`    — derive a deduplicated list of search terms from a
                            natural-language query (token + phrase synonyms +
                            simple plural→singular fallback)
* :func:`score_match`     — score a candidate's fields against the expanded
                            terms and return ``(score, matched_fields,
                            reason)`` — see fields/weights conventions below
* :func:`infer_policy_types` — guess the policy_type slug(s) implied by a
                               free-text query (used as a *fallback* lookup
                               key when direct text search returns nothing)

This module has no SQLAlchemy dependency on purpose — tools call into it after
fetching a candidate set, so it stays trivial to unit-test in isolation.
"""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

# Keep word chars, whitespace, and hyphens so "carry-on" survives.
_PUNCT_RE = re.compile(r"[^\w\s-]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: Optional[str]) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if not text:
        return ""
    out = text.lower().strip()
    out = _PUNCT_RE.sub(" ", out)
    out = _WHITESPACE_RE.sub(" ", out).strip()
    return out


# ---------------------------------------------------------------------------
# Synonym maps
# ---------------------------------------------------------------------------

# Phrase synonyms fire when the multi-word phrase appears as a substring of the
# normalized query. They expand into a curated set of related phrases that
# match how the seed actually wrote about each concept.
_PHRASE_SYNONYMS: dict[str, list[str]] = {
    "damaged packaging": [
        "damaged packaging",
        "opened box",
        "packaging damage",
        "damaged box",
        "broken seal",
        "broken seals",
    ],
    "missing accessories": [
        "missing accessories",
        "incomplete package",
        "missing parts",
        "accessories missing",
        "missing components",
        "incomplete",
    ],
    "delayed flight": [
        "delayed flight",
        "flight delay",
        "late departure",
        "disruption",
        "irregular operation",
        "irregular operations",
        "irrops",
    ],
    "invoice dispute": [
        "invoice dispute",
        "billing dispute",
        "charge dispute",
        "disputed charge",
        "dispute",
    ],
    "overage charge": [
        "overage",
        "overage charge",
        "extra usage",
        "exceeded limit",
        "usage charge",
        "over quota",
    ],
    "checked bag": [
        "checked bag",
        "checked baggage",
        "luggage",
        "hold luggage",
    ],
    "cabin bag": [
        "cabin bag",
        "carry-on",
        "carry on",
        "hand luggage",
    ],
    "open box": [
        "open box",
        "opened box",
        "opened item",
    ],
}

# Word synonyms expand a single token into a curated set of related single- or
# multi-word phrases. The mapping is intentionally directed (e.g. "warranty" →
# "exclusion" but not always the reverse) so noise stays bounded.
_WORD_SYNONYMS: dict[str, list[str]] = {
    "opened": ["opened", "open", "used", "unsealed", "unboxed"],
    "open": ["open", "opened"],
    "unsealed": ["unsealed", "opened"],
    "electronic": ["electronic", "electronics", "device"],
    "electronics": ["electronics", "electronic", "device"],
    "device": ["device", "electronics", "product"],
    "return": ["return", "refund", "exchange"],
    "returns": ["return", "refund", "exchange"],
    "returned": ["return", "returned", "refund"],
    "refund": ["refund", "return", "credit"],
    "refunds": ["refund", "refunds", "credits"],
    "exchange": ["exchange", "return", "swap"],
    "cancellation": ["cancellation", "cancel", "cancelled", "canceled"],
    "cancellations": ["cancellation", "cancellations", "cancel"],
    "cancel": ["cancel", "cancellation", "cancelled", "canceled"],
    "cancelled": ["cancelled", "canceled", "cancellation", "cancel"],
    "canceled": ["canceled", "cancelled", "cancellation", "cancel"],
    "baggage": ["baggage", "luggage", "bag", "checked bag", "cabin bag"],
    "luggage": ["luggage", "baggage", "bag"],
    "bag": ["bag", "baggage", "luggage"],
    "bags": ["bags", "baggage", "luggage"],
    "carry-on": ["carry-on", "carry on", "cabin bag", "hand luggage"],
    "overage": ["overage", "exceeded limit", "extra usage", "usage charge", "over quota"],
    "exceeded": ["exceeded", "overage", "over limit"],
    "warranty": ["warranty", "coverage", "repair", "replacement", "exclusion"],
    "exclusion": ["exclusion", "exclusions", "not covered", "warranty"],
    "exclusions": ["exclusion", "exclusions", "not covered"],
    "damaged": ["damaged", "damage", "broken", "defective"],
    "damage": ["damage", "damaged", "broken"],
    "broken": ["broken", "damaged", "defective"],
    "defective": ["defective", "damaged", "broken"],
    "missing": ["missing", "incomplete", "lacking"],
    "incomplete": ["incomplete", "missing"],
    "accessories": ["accessories", "accessory", "parts"],
    "accessory": ["accessory", "accessories", "parts"],
    "delayed": ["delayed", "delay", "disruption", "late departure"],
    "delay": ["delay", "delayed", "disruption"],
    "disruption": ["disruption", "irregular operations", "irrops", "delay"],
    "flight": ["flight", "departure"],
    "flights": ["flight", "flights"],
    "invoice": ["invoice", "bill", "billing"],
    "invoices": ["invoice", "invoices"],
    "billing": ["billing", "bill", "invoice"],
    "bill": ["bill", "billing", "invoice"],
    "dispute": ["dispute", "complaint", "challenge"],
    "disputed": ["disputed", "dispute"],
    "policy": ["policy"],
    "rules": ["rules", "policy", "rule"],
    "rule": ["rule", "policy"],
    "allowance": ["allowance", "baggage", "limit"],
    "sla": ["sla", "service level", "uptime", "response time"],
    "escalation": ["escalation", "escalate", "priority"],
    "escalate": ["escalate", "escalation"],
    "subscription": ["subscription", "plan", "billing"],
    "outage": ["outage", "incident", "disruption", "downtime"],
    "incident": ["incident", "outage", "disruption"],
    "downtime": ["downtime", "outage", "incident"],
    "hygiene": ["hygiene", "personal care", "sealed"],
    "exception": ["exception", "exclusion", "carve-out"],
    "exceptions": ["exception", "exceptions", "carve-outs"],
}


# Conservative stopwords — short function words that add no retrieval value.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "of", "for", "to", "in", "on", "at", "with", "by",
        "and", "or", "but", "if", "then", "than", "what", "when", "where",
        "why", "how", "who", "whom", "whose", "which", "this", "that", "these",
        "those", "i", "me", "my", "we", "our", "you", "your", "they", "their",
        "can", "could", "should", "would", "will", "may", "might", "must",
        "more", "less", "any", "some", "all", "no", "not", "about", "from",
        "into", "out", "up", "down", "as", "so", "have", "has", "had", "get",
        "got", "very", "really", "please", "tell", "show", "give", "want",
        "need", "let", "us",
    }
)


# ---------------------------------------------------------------------------
# Singularization / token utilities
# ---------------------------------------------------------------------------

# Words whose 's' ending is part of the lemma — never singularize these.
_PLURAL_EXCEPTIONS: frozenset[str] = frozenset(
    {
        # Words whose ``-s``/``-es`` ending is part of the lemma. ``returns`` /
        # ``refunds`` stay plural because the singular form (return/refund) is
        # a separate verb already covered by the word-synonym map.
        "sales", "terms", "exclusions", "rules", "returns", "refunds",
        "exceptions", "operations", "irrops",
        "credits", "accessories", "components", "parts", "bags",
    }
)


def _singularize(word: str) -> str:
    """Cheap, conservative plural-to-singular. Only used as an extra term —
    never replaces the original token."""
    if word in _PLURAL_EXCEPTIONS:
        return word
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("es") and word[-3] in "sxz":
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokens(text: str) -> list[str]:
    """Return content tokens (lowercased, stopwords removed) from text."""
    norm = normalize(text)
    return [t for t in norm.split() if t and t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------


def expand_query(query: Optional[str], *, max_terms: int = 18) -> list[str]:
    """Return a deduplicated, ranked list of search strings for ``query``.

    Order (most → least specific):
      1. the normalized original query (longest, most specific match)
      2. any matched phrase-synonym expansions (multi-word, hand-curated)
      3. each non-stopword token (and its singular form)
      4. word-level synonyms for each token

    The list is capped at ``max_terms`` to keep generated SQL clauses small.
    """
    if not query:
        return []

    norm = normalize(query)
    terms: list[str] = []
    seen: set[str] = set()

    def _add(t: str) -> None:
        t = t.strip()
        if t and t not in seen and len(t) >= 2:
            seen.add(t)
            terms.append(t)

    # 1) Full normalized query.
    _add(norm)

    # 2) Phrase synonyms (longest phrase first so we never expand a sub-phrase
    #    that's already covered by a longer match).
    for phrase in sorted(_PHRASE_SYNONYMS, key=len, reverse=True):
        if phrase in norm:
            for syn in _PHRASE_SYNONYMS[phrase]:
                _add(syn)

    # 3 + 4) Token-level expansion.
    for tok in _tokens(query):
        _add(tok)
        sing = _singularize(tok)
        if sing != tok:
            _add(sing)
        for syn in _WORD_SYNONYMS.get(tok, []):
            _add(syn)
        if sing != tok:
            for syn in _WORD_SYNONYMS.get(sing, []):
                _add(syn)

    return terms[:max_terms]


# ---------------------------------------------------------------------------
# Match scoring
# ---------------------------------------------------------------------------

# Default per-field weights. Title-like fields are weighted higher than body
# because a hit there is more semantically specific. ``key`` fields (slugs,
# clause keys, policy types) are also weighted high — they're how the seed
# encodes the conceptual category.
_DEFAULT_WEIGHTS: dict[str, float] = {
    "title": 3.0,
    "rule_name": 3.0,
    "name": 3.0,
    "clause_key": 2.5,
    "policy_type": 2.5,
    "incident_type": 2.5,
    "category": 2.5,
    "note_type": 2.0,
    "product_category_name": 2.0,
    "applies_to": 1.5,
    "exceptions": 1.5,
    "body": 1.0,
    "excerpt": 1.0,
}


def score_match(
    terms: list[str],
    fields: dict[str, Optional[str]],
    *,
    weights: Optional[dict[str, float]] = None,
) -> tuple[float, list[str], str]:
    """Score ``fields`` against the expanded ``terms``.

    Returns ``(match_score, matched_fields, reason)``:

    * ``match_score`` — normalized into ``[0, 1]``: each (term, field) hit
      contributes the field's weight; the score is divided by the maximum
      hypothetical score (best term × every weighted field). Multiple terms
      may hit the same field; each hit counts.
    * ``matched_fields`` — fields that contributed at least one hit, in the
      order they first matched.
    * ``reason`` — a short human-readable sentence ("matched 'opened' in
      body; 'electronics' in category"), useful for trace explanation.
    """
    weights = weights or _DEFAULT_WEIGHTS
    if not terms:
        return 0.0, [], "no query terms"

    matched_fields: list[str] = []
    seen_fields: set[str] = set()
    score = 0.0
    # For the headline reason, prefer the longest term hit on the highest-
    # weighted field.
    best_score = -1.0
    best_term = ""
    best_field = ""
    field_to_top_term: dict[str, str] = {}

    for field_name, field_text in fields.items():
        if not field_text:
            continue
        lc = field_text.lower()
        w = weights.get(field_name, 0.5)
        for term in terms:
            if not term or term not in lc:
                continue
            score += w
            if field_name not in seen_fields:
                seen_fields.add(field_name)
                matched_fields.append(field_name)
            # Headline: prioritise high-weight fields, then longer terms.
            this = w * 10 + len(term)
            if this > best_score:
                best_score = this
                best_term = term
                best_field = field_name
            if field_name not in field_to_top_term or len(term) > len(
                field_to_top_term[field_name]
            ):
                field_to_top_term[field_name] = term

    if not matched_fields:
        return 0.0, [], "no direct term match (fallback candidate)"

    # Normalize: divide by max possible (all weighted fields × number of terms).
    max_possible = sum(weights.get(f, 0.5) for f in fields if fields.get(f)) * len(terms)
    norm_score = min(1.0, score / max_possible) if max_possible > 0 else 0.0

    # Pretty reason: list up to 3 field hits with their top term.
    parts = [
        f"'{field_to_top_term[f]}' in {f}"
        for f in matched_fields[:3]
    ]
    reason = "matched " + "; ".join(parts)
    return norm_score, matched_fields, reason


# ---------------------------------------------------------------------------
# Policy-type inference (used by tools as a fallback lookup)
# ---------------------------------------------------------------------------

# Maps query keywords to seeded ``policy_documents.policy_type`` slugs.
_POLICY_TYPE_HINTS: dict[str, list[str]] = {
    "refund_policy": [
        "refund", "refunds", "money back", "money-back",
        # Delayed/disrupted flights are typically addressed by the airline's
        # refund and cancellation policies (eligibility / rebooking).
        "delay", "delayed", "irrops", "disruption",
    ],
    "cancellation_policy": [
        "cancel", "cancellation", "cancelled", "canceled",
        "delay", "delayed", "missed connection",
    ],
    "baggage_policy": [
        "baggage", "luggage", "bag", "checked", "carry-on", "carry on",
        "allowance",
    ],
    "return_policy": ["return", "returns", "exchange", "restocking"],
    "warranty_policy": [
        "warranty", "coverage", "exclusion", "exclusions",
        # The airline 'warranty_policy' (Service Guarantee) also covers on-time
        # performance and >3-hour delay vouchers.
        "service guarantee", "on-time", "on time",
    ],
    "overage_policy": [
        "overage", "exceeded", "over quota", "extra usage", "usage charge",
    ],
    "escalation_policy": ["escalation", "escalate", "sla", "priority"],
    "subscription_policy": ["subscription", "downgrade", "upgrade"],
    "payment_policy": ["payment", "billing", "invoice", "dispute"],
    "privacy_policy": ["privacy", "data retention", "gdpr", "dpa"],
}


def infer_policy_types(query: Optional[str]) -> list[str]:
    """Best-effort guess at which ``policy_type`` slug(s) the query is about.

    Used by tools as a *fallback* lookup key when free-text retrieval returns
    nothing. Returns a possibly-empty list ordered by specificity (longest
    matching keyword first).
    """
    norm = normalize(query)
    if not norm:
        return []

    hits: list[tuple[int, str]] = []
    for slug, keywords in _POLICY_TYPE_HINTS.items():
        for kw in keywords:
            if kw in norm:
                hits.append((len(kw), slug))
                break
    hits.sort(key=lambda x: -x[0])
    out: list[str] = []
    for _, slug in hits:
        if slug not in out:
            out.append(slug)
    return out


# ---------------------------------------------------------------------------
# Tiny helpers re-used across tools
# ---------------------------------------------------------------------------


def make_excerpt(text: Optional[str], max_len: int = 240) -> str:
    """Whitespace-normalized, length-capped excerpt with an ellipsis."""
    if not text:
        return ""
    out = " ".join(text.split())
    return out if len(out) <= max_len else out[: max_len - 1].rstrip() + "…"
