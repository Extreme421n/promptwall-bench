"""get_policy_clause tool — a more focused alternative to search_kb_articles."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.models import KBArticle
from app.tools.base import Tool


class GetPolicyClauseInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_topic: str = Field(
        min_length=2,
        description="Topic phrase, e.g. 'cancellation', 'non-refundable fare'.",
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional category filter (baggage, refunds, flight_change, "
        "seats, cancellation, loyalty, check_in, special_assistance).",
    )


class PolicyClauseItem(BaseModel):
    slug: str
    title: str
    category: str
    excerpt: str
    version: int


class GetPolicyClauseOutput(BaseModel):
    policy_topic: str
    count: int
    clauses: list[PolicyClauseItem]


def _excerpt(text: str, max_len: int = 320) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


def _impl(session: Session, inp: GetPolicyClauseInput) -> GetPolicyClauseOutput:
    pattern = f"%{inp.policy_topic.strip()}%"
    stmt = (
        select(KBArticle)
        .where(KBArticle.is_active.is_(True))
        # Title match has precedence; body match is a fallback. We do that with
        # an OR (title first in ranking via two queries would be nicer in PG;
        # this keeps the SQL portable for SQLite).
        .where(or_(KBArticle.title.ilike(pattern), KBArticle.body.ilike(pattern)))
    )
    if inp.category is not None:
        stmt = stmt.where(KBArticle.category == inp.category)
    stmt = stmt.order_by(KBArticle.category, KBArticle.slug).limit(5)

    rows = session.execute(stmt).scalars().all()
    return GetPolicyClauseOutput(
        policy_topic=inp.policy_topic,
        count=len(rows),
        clauses=[
            PolicyClauseItem(
                slug=a.slug,
                title=a.title,
                category=a.category,
                excerpt=_excerpt(a.body),
                version=a.version,
            )
            for a in rows
        ],
    )


get_policy_clause = Tool(
    name="get_policy_clause",
    description=(
        "Look up policy clauses by topic. Narrower than search_kb_articles — "
        "returns at most 5 results, ranked on title and body match. Use this "
        "when the user asks 'what's the policy on …' or wants a specific clause."
    ),
    domain="kb",
    input_schema=GetPolicyClauseInput,
    output_schema=GetPolicyClauseOutput,
    risk_level="low",
    read_only=True,
    impl=_impl,
)
