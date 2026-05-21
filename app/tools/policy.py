"""get_policy_clause tool — queries the structured policy_clauses table.

Phase 6B-4 update: this tool now queries the dedicated policy_clauses table
(Phase 6B-1) instead of the KB articles. Old callers that passed
``policy_topic`` continue to work via an accepted alias.

Phase 6C-1 update: queries are normalized + synonym-expanded, search now
covers clause title, body, clause_key, applies_to, exceptions, *and* the parent
document's title and policy_type. When direct text search returns nothing, the
tool falls back to a policy-type-inferred lookup so realistic wording like
"cancellation rules" still finds the cancellation_policy clauses.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import and_, case, or_, select
from sqlalchemy.orm import Session

from app.models import PolicyClause, PolicyDocument
from app.tools._text_search import (
    expand_query,
    infer_policy_types,
    make_excerpt,
    score_match,
)
from app.tools.base import Tool


class GetPolicyClauseInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: Optional[str] = Field(
        default=None,
        description="Free-text match against clause title and body.",
    )
    domain: Optional[str] = Field(
        default=None,
        description="Optional domain filter (airline / commerce / saas / support / crm).",
    )
    policy_type: Optional[str] = Field(
        default=None,
        description="Optional policy_type filter (refund_policy, baggage_policy, etc.).",
    )
    clause_key: Optional[str] = Field(
        default=None,
        description="Match an exact clause_key (e.g. 'eligibility', 'grace_period').",
    )
    # Back-compat alias for callers that used the pre-6B-4 schema.
    policy_topic: Optional[str] = Field(
        default=None,
        description="DEPRECATED — alias for ``query``. Use ``query`` going forward.",
    )

    @model_validator(mode="after")
    def _at_least_one(self) -> "GetPolicyClauseInput":
        if self.policy_topic and not self.query:
            self.query = self.policy_topic
        if not any((self.query, self.domain, self.policy_type, self.clause_key)):
            raise ValueError(
                "provide at least one of query, domain, policy_type, clause_key"
            )
        return self


class PolicyClauseItem(BaseModel):
    clause_id: int
    policy_document_id: int
    policy_title: str
    policy_domain: str
    policy_type: str
    clause_key: str
    title: str
    body: str
    severity: str
    applies_to: Optional[str]
    exceptions: Optional[str]
    # Phase 6C-1 — retrieval-explanation fields.
    excerpt: str = ""
    match_score: float = 0.0
    match_reason: str = ""
    matched_fields: list[str] = Field(default_factory=list)


class GetPolicyClauseOutput(BaseModel):
    count: int
    clauses: list[PolicyClauseItem]
    query_terms: list[str] = Field(default_factory=list)
    fallback_used: bool = False
    inferred_policy_types: list[str] = Field(default_factory=list)


def _build_item(
    clause: PolicyClause,
    doc: PolicyDocument,
    terms: list[str],
    inferred: list[str] | None = None,
) -> PolicyClauseItem:
    if terms:
        score, fields, reason = score_match(
            terms,
            {
                "title": clause.title,
                "clause_key": clause.clause_key,
                "policy_type": doc.policy_type,
                "applies_to": clause.applies_to,
                "exceptions": clause.exceptions,
                "body": clause.body,
            },
        )
        # Policy-type boost: if the user's free-text query implies a specific
        # policy_type slug (e.g. "cancellation rules" → cancellation_policy),
        # rows in that policy_type are *much* more likely to be relevant.
        if inferred and doc.policy_type in inferred:
            score = min(1.0, score * 2.0 + 0.05)
            if "policy_type" not in fields:
                fields = ["policy_type"] + fields
            reason = f"policy_type={doc.policy_type} matches inferred query intent; {reason}"
    else:
        score, fields, reason = (1.0, ["filter"], "filter-only lookup")
    return PolicyClauseItem(
        clause_id=clause.id,
        policy_document_id=clause.policy_document_id,
        policy_title=doc.title,
        policy_domain=doc.domain,
        policy_type=doc.policy_type,
        clause_key=clause.clause_key,
        title=clause.title,
        body=clause.body,
        severity=clause.severity,
        applies_to=clause.applies_to,
        exceptions=clause.exceptions,
        excerpt=make_excerpt(clause.body),
        match_score=score,
        match_reason=reason,
        matched_fields=fields,
    )


def _impl(session: Session, inp: GetPolicyClauseInput) -> GetPolicyClauseOutput:
    terms = expand_query(inp.query) if inp.query else []
    inferred = infer_policy_types(inp.query) if inp.query else []

    base_filters = [PolicyDocument.is_active.is_(True)]
    if inp.domain:
        base_filters.append(PolicyDocument.domain == inp.domain.strip())
    if inp.policy_type:
        base_filters.append(PolicyDocument.policy_type == inp.policy_type.strip())
    if inp.clause_key:
        base_filters.append(PolicyClause.clause_key == inp.clause_key.strip())

    primary_filters = list(base_filters)
    if terms:
        from sqlalchemy.sql import or_ as _or_

        def _orcol(col):
            return _or_(*[col.ilike(f"%{t}%") for t in terms])

        # Match terms only against the CLAUSE's own fields. Document-level
        # title / policy_type matching is intentionally NOT used here — it
        # would surface every clause inside a doc whose title or slug happens
        # to contain the query word (e.g. an unrelated "service credits"
        # clause inside the refund_policy doc). The inferred policy_type list
        # below is used to *order* and *boost* the candidate window instead.
        text_or = or_(
            _orcol(PolicyClause.title),
            _orcol(PolicyClause.body),
            _orcol(PolicyClause.clause_key),
            _orcol(PolicyClause.applies_to),
            _orcol(PolicyClause.exceptions),
        )
        primary_filters.append(text_or)

    # If we have inferred policy_types, sort those rows FIRST in the SQL
    # candidate window so they survive the LIMIT cut. Without this, an
    # alphabetically-early policy_type (e.g. baggage_policy) can fill the
    # whole window and shadow the inferred-type rows we actually want to
    # re-rank to the top.
    if inferred:
        type_priority = case(
            (PolicyDocument.policy_type.in_(inferred), 0),
            else_=1,
        )
        ordering = [
            type_priority,
            PolicyDocument.domain,
            PolicyDocument.policy_type,
            PolicyClause.id,
        ]
    else:
        ordering = [
            PolicyDocument.domain,
            PolicyDocument.policy_type,
            PolicyClause.id,
        ]

    stmt = (
        select(PolicyClause, PolicyDocument)
        .join(PolicyDocument, PolicyClause.policy_document_id == PolicyDocument.id)
        .where(and_(*primary_filters))
        # Pull a wide candidate window — the Python re-rank does the real work.
        .order_by(*ordering)
        .limit(120)
    )
    rows = session.execute(stmt).all()

    fallback_used = False
    if not rows and inferred:
        # Fallback: drop the free-text filter and use the inferred policy_type
        # slug(s) instead. Keeps domain/clause_key filters if the caller set them.
        fallback_filters = list(base_filters)
        if not inp.policy_type:
            fallback_filters.append(PolicyDocument.policy_type.in_(inferred))
        rows = session.execute(
            select(PolicyClause, PolicyDocument)
            .join(PolicyDocument, PolicyClause.policy_document_id == PolicyDocument.id)
            .where(and_(*fallback_filters))
            .order_by(PolicyDocument.domain, PolicyDocument.policy_type, PolicyClause.id)
            .limit(20)
        ).all()
        fallback_used = bool(rows)

    items = [_build_item(c, d, terms, inferred) for c, d in rows]
    items.sort(key=lambda x: (-x.match_score, x.clause_id))
    items = items[:10]
    return GetPolicyClauseOutput(
        count=len(items),
        clauses=items,
        query_terms=terms,
        fallback_used=fallback_used,
        inferred_policy_types=inferred,
    )


get_policy_clause = Tool(
    name="get_policy_clause",
    description=(
        "Look up policy clauses by any combination of: query (free-text on "
        "title/body/clause_key/applies_to/exceptions, plus the parent "
        "document's title and policy_type — synonym-expanded), domain, "
        "policy_type, clause_key. Falls back to an inferred policy_type "
        "lookup when direct text search returns nothing. Active policies only."
    ),
    domain="kb",
    input_schema=GetPolicyClauseInput,
    output_schema=GetPolicyClauseOutput,
    risk_level="low",
    read_only=True,
    impl=_impl,
)
