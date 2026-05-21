"""search_policy_documents + get_latest_policy_version tools.

Phase 6B-4 update: ``search_policy_documents`` now queries the structured
``policy_documents`` table (Phase 6B-1) rather than the KB articles. The
``get_latest_policy_version`` tool (Phase C2) still queries KB articles for
backwards compatibility with existing eval cases.
"""

from __future__ import annotations

from datetime import date as _date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import and_, case, or_, select
from sqlalchemy.orm import Session

from app.models import KBArticle, PolicyDocument
from app.tools._text_search import (
    expand_query,
    infer_policy_types,
    make_excerpt,
    score_match,
)
from app.tools.base import ResourceNotFoundError, Tool


# ---------------------------------------------------------------------------
# search_policy_documents — queries policy_documents (Phase 6B-1)
# ---------------------------------------------------------------------------


class SearchPolicyDocumentsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, description="Free-text match on title and body.")
    domain: Optional[str] = Field(
        default=None,
        description="Optional domain filter (airline / commerce / saas / support / crm).",
    )
    policy_type: Optional[str] = Field(
        default=None,
        description="Optional policy_type filter (refund_policy / return_policy / …).",
    )
    limit: int = Field(default=10, ge=1, le=25)


class PolicyDocumentItem(BaseModel):
    id: int
    title: str
    domain: str
    policy_type: str
    version: int
    excerpt: str
    effective_from: _date
    effective_to: Optional[_date]
    is_active: bool
    # Phase 6C-1 — retrieval-explanation fields.
    match_score: float = 0.0
    match_reason: str = ""
    matched_fields: list[str] = Field(default_factory=list)


class SearchPolicyDocumentsOutput(BaseModel):
    count: int
    documents: list[PolicyDocumentItem]
    query_terms: list[str] = Field(default_factory=list)
    fallback_used: bool = False
    inferred_policy_types: list[str] = Field(default_factory=list)


def _excerpt(text: str, max_len: int = 240) -> str:
    """Back-compat shim; ``make_excerpt`` is the new canonical helper."""
    return make_excerpt(text, max_len=max_len)


def _build_doc_item(
    d: PolicyDocument, terms: list[str], inferred: list[str] | None = None
) -> PolicyDocumentItem:
    if terms:
        score, fields, reason = score_match(
            terms,
            {
                "title": d.title,
                "policy_type": d.policy_type,
                "body": d.body,
            },
        )
        if inferred and d.policy_type in inferred:
            score = min(1.0, score * 2.0 + 0.05)
            if "policy_type" not in fields:
                fields = ["policy_type"] + fields
            reason = f"policy_type={d.policy_type} matches inferred query intent; {reason}"
    else:
        score, fields, reason = (1.0, ["filter"], "filter-only lookup")
    return PolicyDocumentItem(
        id=d.id,
        title=d.title,
        domain=d.domain,
        policy_type=d.policy_type,
        version=d.version,
        excerpt=make_excerpt(d.body),
        effective_from=d.effective_from,
        effective_to=d.effective_to,
        is_active=d.is_active,
        match_score=score,
        match_reason=reason,
        matched_fields=fields,
    )


def _search_policy_documents_impl(
    session: Session, inp: SearchPolicyDocumentsInput
) -> SearchPolicyDocumentsOutput:
    terms = expand_query(inp.query)
    inferred = infer_policy_types(inp.query)

    base_filters: list = [PolicyDocument.is_active.is_(True)]
    if inp.domain:
        base_filters.append(PolicyDocument.domain == inp.domain.strip())
    if inp.policy_type:
        base_filters.append(PolicyDocument.policy_type == inp.policy_type.strip())

    primary_filters = list(base_filters)
    if terms:
        text_or = or_(
            or_(*[PolicyDocument.title.ilike(f"%{t}%") for t in terms]),
            or_(*[PolicyDocument.body.ilike(f"%{t}%") for t in terms]),
            or_(*[PolicyDocument.policy_type.ilike(f"%{t}%") for t in terms]),
        )
        primary_filters.append(text_or)

    # Push inferred policy_type rows to the front of the SQL candidate window.
    if inferred:
        type_priority = case(
            (PolicyDocument.policy_type.in_(inferred), 0),
            else_=1,
        )
        ordering = [
            type_priority,
            PolicyDocument.domain,
            PolicyDocument.policy_type,
            PolicyDocument.version.desc(),
        ]
    else:
        ordering = [
            PolicyDocument.domain,
            PolicyDocument.policy_type,
            PolicyDocument.version.desc(),
        ]

    stmt = (
        select(PolicyDocument)
        .where(and_(*primary_filters))
        .order_by(*ordering)
        # Pull a wider candidate window so the Python re-rank has room.
        .limit(max(inp.limit * 6, 60))
    )
    rows = list(session.execute(stmt).scalars().all())

    fallback_used = False
    if not rows and inferred:
        # Fallback: drop the free-text filter and use the inferred policy_type
        # slug(s) instead. Keeps any domain filter the caller already passed.
        fb = list(base_filters)
        if not inp.policy_type:
            fb.append(PolicyDocument.policy_type.in_(inferred))
        rows = list(
            session.execute(
                select(PolicyDocument)
                .where(and_(*fb))
                .order_by(PolicyDocument.version.desc(), PolicyDocument.id)
                .limit(inp.limit)
            )
            .scalars()
            .all()
        )
        fallback_used = bool(rows)

    items = [_build_doc_item(d, terms, inferred) for d in rows]
    items.sort(key=lambda x: (-x.match_score, x.id))
    items = items[: inp.limit]
    return SearchPolicyDocumentsOutput(
        count=len(items),
        documents=items,
        query_terms=terms,
        fallback_used=fallback_used,
        inferred_policy_types=inferred,
    )


search_policy_documents = Tool(
    name="search_policy_documents",
    description=(
        "Search active policy documents from the structured policy_documents "
        "table by free-text on title/body, optionally narrowed by domain "
        "(airline/commerce/saas/support/crm) and policy_type (refund_policy, "
        "baggage_policy, etc.). Returns up to 25 results with excerpts."
    ),
    domain="kb",
    input_schema=SearchPolicyDocumentsInput,
    output_schema=SearchPolicyDocumentsOutput,
    risk_level="low",
    read_only=True,
    impl=_search_policy_documents_impl,
)


# ---------------------------------------------------------------------------
# get_latest_policy_version — unchanged from Phase C2 (queries KBArticle)
# ---------------------------------------------------------------------------


class GetLatestPolicyVersionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slug: str = Field(
        min_length=2,
        description="The KB article slug, e.g. 'baggage-1' or 'refunds-2'.",
    )


class GetLatestPolicyVersionOutput(BaseModel):
    slug: str
    title: str
    category: str
    version: int
    is_active: bool
    body_excerpt: str
    updated_at: datetime


def _latest_version_impl(
    session: Session, inp: GetLatestPolicyVersionInput
) -> GetLatestPolicyVersionOutput:
    article = session.execute(
        select(KBArticle)
        .where(KBArticle.slug == inp.slug.strip())
        .order_by(KBArticle.version.desc(), KBArticle.id.desc())
        .limit(1)
    ).scalar_one_or_none()
    if article is None:
        raise ResourceNotFoundError(f"policy {inp.slug!r} not found")
    return GetLatestPolicyVersionOutput(
        slug=article.slug,
        title=article.title,
        category=article.category,
        version=article.version,
        is_active=article.is_active,
        body_excerpt=_excerpt(article.body, max_len=600),
        updated_at=article.updated_at,
    )


get_latest_policy_version = Tool(
    name="get_latest_policy_version",
    description=(
        "Return the latest version of a KB article by slug. Useful when a "
        "user references a specific KB slug and you need the current copy."
    ),
    domain="kb",
    input_schema=GetLatestPolicyVersionInput,
    output_schema=GetLatestPolicyVersionOutput,
    risk_level="low",
    read_only=True,
    impl=_latest_version_impl,
)
