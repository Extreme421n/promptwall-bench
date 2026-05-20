"""search_kb_articles tool."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.models import KBArticle
from app.tools.base import Tool


class SearchKbArticlesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, description="Free-text search across title and body")
    category: Optional[str] = Field(
        default=None,
        description="Optional category filter: baggage, refunds, flight_change, "
        "seats, cancellation, loyalty, check_in, special_assistance",
    )
    limit: int = Field(default=10, ge=1, le=25, description="Maximum results to return")


class KBArticleItem(BaseModel):
    slug: str
    title: str
    category: str
    excerpt: str
    version: int


class SearchKbArticlesOutput(BaseModel):
    count: int
    articles: list[KBArticleItem]


def _excerpt(text: str, max_len: int = 200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


def _impl(session: Session, inp: SearchKbArticlesInput) -> SearchKbArticlesOutput:
    pattern = f"%{inp.query.strip()}%"
    stmt = (
        select(KBArticle)
        .where(KBArticle.is_active.is_(True))
        .where(or_(KBArticle.title.ilike(pattern), KBArticle.body.ilike(pattern)))
    )
    if inp.category is not None:
        stmt = stmt.where(KBArticle.category == inp.category)
    stmt = stmt.order_by(KBArticle.category, KBArticle.slug).limit(inp.limit)

    rows = session.execute(stmt).scalars().all()
    items = [
        KBArticleItem(
            slug=a.slug,
            title=a.title,
            category=a.category,
            excerpt=_excerpt(a.body),
            version=a.version,
        )
        for a in rows
    ]
    return SearchKbArticlesOutput(count=len(items), articles=items)


search_kb_articles = Tool(
    name="search_kb_articles",
    description=(
        "Search active knowledge-base articles by free-text query (matched "
        "against title and body), optionally filtered by category. Returns up "
        "to 25 results with a short excerpt."
    ),
    domain="kb",
    input_schema=SearchKbArticlesInput,
    output_schema=SearchKbArticlesOutput,
    risk_level="low",
    read_only=True,
    impl=_impl,
)
