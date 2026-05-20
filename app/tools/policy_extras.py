"""Phase C2 KB / policy tools.

* search_policy_documents      — narrower than search_kb_articles; restricted
                                  to the policy-style categories and returns
                                  versioned metadata.
* get_latest_policy_version    — given a slug, return the current version row.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.models import KBArticle
from app.tools.base import ResourceNotFoundError, Tool


# Categories that are policy-document-shaped (vs. how-to / FAQ).
_POLICY_CATEGORIES = (
    "baggage",
    "refunds",
    "flight_change",
    "cancellation",
    "loyalty",
    "special_assistance",
)


# ---------------------------------------------------------------------------
# search_policy_documents
# ---------------------------------------------------------------------------


class SearchPolicyDocumentsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2)
    category: Optional[str] = Field(
        default=None,
        description=(
            "Optional category filter. One of: " + ", ".join(_POLICY_CATEGORIES)
        ),
    )
    limit: int = Field(default=10, ge=1, le=25)


class PolicyDocumentItem(BaseModel):
    slug: str
    title: str
    category: str
    excerpt: str
    version: int
    is_active: bool
    updated_at: datetime


class SearchPolicyDocumentsOutput(BaseModel):
    count: int
    documents: list[PolicyDocumentItem]


def _excerpt(text: str, max_len: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


def _search_policy_impl(
    session: Session, inp: SearchPolicyDocumentsInput
) -> SearchPolicyDocumentsOutput:
    pattern = f"%{inp.query.strip()}%"
    stmt = (
        select(KBArticle)
        .where(KBArticle.is_active.is_(True))
        .where(KBArticle.category.in_(_POLICY_CATEGORIES))
        .where(or_(KBArticle.title.ilike(pattern), KBArticle.body.ilike(pattern)))
    )
    if inp.category is not None:
        if inp.category not in _POLICY_CATEGORIES:
            # An explicit category outside the policy set returns no rows
            # rather than 404 — the chatbot may have guessed.
            return SearchPolicyDocumentsOutput(count=0, documents=[])
        stmt = stmt.where(KBArticle.category == inp.category)
    stmt = stmt.order_by(KBArticle.category, KBArticle.slug).limit(inp.limit)

    rows = session.execute(stmt).scalars().all()
    return SearchPolicyDocumentsOutput(
        count=len(rows),
        documents=[
            PolicyDocumentItem(
                slug=a.slug,
                title=a.title,
                category=a.category,
                excerpt=_excerpt(a.body),
                version=a.version,
                is_active=a.is_active,
                updated_at=a.updated_at,
            )
            for a in rows
        ],
    )


search_policy_documents = Tool(
    name="search_policy_documents",
    description=(
        "Search active POLICY documents (a strict subset of the KB: baggage, "
        "refunds, flight_change, cancellation, loyalty, special_assistance). "
        "Returns versioned policy metadata alongside the matching excerpt. "
        "Prefer this over search_kb_articles when the user is asking about a "
        "specific policy."
    ),
    domain="kb",
    input_schema=SearchPolicyDocumentsInput,
    output_schema=SearchPolicyDocumentsOutput,
    risk_level="low",
    read_only=True,
    impl=_search_policy_impl,
)


# ---------------------------------------------------------------------------
# get_latest_policy_version
# ---------------------------------------------------------------------------


class GetLatestPolicyVersionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slug: str = Field(
        min_length=2,
        description="The policy article slug, e.g. 'baggage-1' or 'refunds-2'.",
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
        "Return the latest version of a policy document by slug. Useful when "
        "the user references a specific policy and you need the authoritative "
        "current copy."
    ),
    domain="kb",
    input_schema=GetLatestPolicyVersionInput,
    output_schema=GetLatestPolicyVersionOutput,
    risk_level="low",
    read_only=True,
    impl=_latest_version_impl,
)
