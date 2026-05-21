"""Phase 6B-4 textual-retrieval tools, with Phase 6C-1 retrieval-quality upgrade.

Read-only queries over the Phase 6B-1 textual knowledge tables. All tools log
through the existing ToolExecutor (registered into ``default_registry`` from
``app.tools.__init__``).

Phase 6C-1 improvements (no LLM; pure deterministic helpers in
``app.tools._text_search``):
    * Query normalization (lowercase, punctuation, whitespace)
    * Synonym expansion (phrase + word level — e.g. "opened" → open, used,
      unsealed; "delayed flight" → flight delay, late departure, IRROPS)
    * Multi-field search (titles, bodies, exceptions, applies_to,
      category names, clause keys, policy types)
    * Best-effort policy-type fallback when free-text search returns nothing
    * Every item carries ``match_score``, ``match_reason``, ``matched_fields``,
      and ``excerpt`` so the chatbot / scorer can explain *why* a row matched

* search_return_rules               — find product-return rules by category + text
* get_product_warranty_terms        — warranty terms for a product (by sku or id)
* search_internal_agent_notes       — agent notes for a customer + optional text
* search_operational_incidents      — search incidents by domain / text / active state
* get_support_resolution_template   — fetch a support template by category + text
* list_policy_versions              — every version of a (domain, policy_type) policy
* get_active_policy                 — currently-active policy for a (domain, policy_type)
"""

from __future__ import annotations

from datetime import date as _date, datetime
from decimal import Decimal
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from app.models import (
    Customer,
    InternalAgentNote,
    OperationalIncident,
    PolicyDocument,
    Product,
    ProductCategory,
    ProductReturnRule,
    ProductWarrantyTerms,
    SupportResolutionTemplate,
)
from app.tools._text_search import (
    expand_query,
    infer_policy_types,
    make_excerpt,
    score_match,
)
from app.tools.base import ResourceNotFoundError, Tool


# Backwards-compat alias for the small handful of callers still importing the
# old private helper. New code should use ``make_excerpt`` directly.
def _excerpt(text: Optional[str], max_len: int = 240) -> str:
    return make_excerpt(text, max_len=max_len)


def _ilike_or(column, terms: list[str]):
    """Build an OR-clause of ``column ILIKE %term%`` over many terms.

    Returns ``None`` if ``terms`` is empty so callers can skip the filter.
    """
    if not terms:
        return None
    return or_(*[column.ilike(f"%{t}%") for t in terms])


# ---------------------------------------------------------------------------
# 1. search_return_rules
# ---------------------------------------------------------------------------


class SearchReturnRulesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, description="Free-text match across rule name, body, exceptions, and category.")
    product_category: Optional[str] = Field(
        default=None,
        description="Optional category name (partial match, case-insensitive).",
    )
    limit: int = Field(default=10, ge=1, le=25)


class ReturnRuleItem(BaseModel):
    id: int
    product_category_id: int
    product_category_name: str
    rule_name: str
    body_excerpt: str
    opened_item_allowed: bool
    return_window_days: int
    restocking_fee_percent: Decimal
    exceptions: Optional[str]
    # Phase 6C-1 — retrieval-explanation fields.
    excerpt: str = ""
    match_score: float = 0.0
    match_reason: str = ""
    matched_fields: list[str] = Field(default_factory=list)


class SearchReturnRulesOutput(BaseModel):
    count: int
    rules: list[ReturnRuleItem]
    query_terms: list[str] = Field(default_factory=list)
    fallback_used: bool = False


def _build_return_rule_item(
    r: ProductReturnRule, cat_name: str, terms: list[str]
) -> ReturnRuleItem:
    body_excerpt = make_excerpt(r.body)
    score, fields, reason = score_match(
        terms,
        {
            "rule_name": r.rule_name,
            "product_category_name": cat_name,
            "body": r.body,
            "exceptions": r.exceptions,
        },
    )
    return ReturnRuleItem(
        id=r.id,
        product_category_id=r.product_category_id,
        product_category_name=cat_name,
        rule_name=r.rule_name,
        body_excerpt=body_excerpt,
        opened_item_allowed=r.opened_item_allowed,
        return_window_days=r.return_window_days,
        restocking_fee_percent=r.restocking_fee_percent,
        exceptions=r.exceptions,
        excerpt=body_excerpt,
        match_score=score,
        match_reason=reason,
        matched_fields=fields,
    )


def _search_return_rules_impl(
    session: Session, inp: SearchReturnRulesInput
) -> SearchReturnRulesOutput:
    terms = expand_query(inp.query)

    # Primary: OR across rule_name, body, exceptions, category name.
    rule_or = or_(
        _ilike_or(ProductReturnRule.rule_name, terms),
        _ilike_or(ProductReturnRule.body, terms),
        _ilike_or(ProductReturnRule.exceptions, terms),
        _ilike_or(ProductCategory.name, terms),
    )
    filters = [rule_or] if terms else []
    if inp.product_category:
        filters.append(ProductCategory.name.ilike(f"%{inp.product_category.strip()}%"))

    stmt = (
        select(ProductReturnRule, ProductCategory.name)
        .join(
            ProductCategory,
            ProductReturnRule.product_category_id == ProductCategory.id,
        )
        .where(and_(*filters) if filters else True)
        .order_by(ProductReturnRule.id)
        # Pull a larger candidate window than ``limit`` so we can re-rank in
        # Python and still respect ``limit``.
        .limit(max(inp.limit * 4, 40))
    )
    rows = session.execute(stmt).all()

    fallback_used = False
    if not rows and inp.product_category:
        # Fallback 1: drop the text search, keep the category filter.
        fallback_used = True
        rows = session.execute(
            select(ProductReturnRule, ProductCategory.name)
            .join(
                ProductCategory,
                ProductReturnRule.product_category_id == ProductCategory.id,
            )
            .where(ProductCategory.name.ilike(f"%{inp.product_category.strip()}%"))
            .order_by(ProductReturnRule.id)
            .limit(inp.limit)
        ).all()
    elif not rows:
        # Fallback 2: try to match a category name against any token of the
        # original query — e.g. "electronics" / "apparel" / "kitchen".
        tokens_only = [t for t in terms if " " not in t]
        category_or = _ilike_or(ProductCategory.name, tokens_only)
        if category_or is not None:
            fallback_used = True
            rows = session.execute(
                select(ProductReturnRule, ProductCategory.name)
                .join(
                    ProductCategory,
                    ProductReturnRule.product_category_id == ProductCategory.id,
                )
                .where(category_or)
                .order_by(ProductReturnRule.id)
                .limit(inp.limit)
            ).all()

    scored = [_build_return_rule_item(r, cat, terms) for r, cat in rows]
    scored.sort(key=lambda x: (-x.match_score, x.id))
    items = scored[: inp.limit]
    return SearchReturnRulesOutput(
        count=len(items),
        rules=items,
        query_terms=terms,
        fallback_used=fallback_used,
    )


search_return_rules = Tool(
    name="search_return_rules",
    description=(
        "Search product-return rules by free-text on rule_name, body, "
        "exceptions, and product category. Synonym-expands the query (e.g. "
        "'opened electronic' also matches 'open', 'used', 'unsealed', "
        "'electronics', 'device'). Returns ranked rules with return window, "
        "restocking fee, exceptions, and a match_reason."
    ),
    domain="commerce",
    input_schema=SearchReturnRulesInput,
    output_schema=SearchReturnRulesOutput,
    risk_level="low",
    read_only=True,
    impl=_search_return_rules_impl,
)


# ---------------------------------------------------------------------------
# 2. get_product_warranty_terms
# ---------------------------------------------------------------------------


class GetProductWarrantyTermsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku: Optional[str] = Field(default=None)
    product_id: Optional[int] = Field(default=None)

    @model_validator(mode="after")
    def _exactly_one(self) -> "GetProductWarrantyTermsInput":
        provided = sum(x is not None for x in (self.sku, self.product_id))
        if provided != 1:
            raise ValueError("provide exactly one of sku, product_id")
        return self


class WarrantyTermsItem(BaseModel):
    warranty_id: int
    product_id: int
    product_name: str
    product_sku: str
    warranty_type: str
    duration_months: int
    body: str
    exclusions: Optional[str]
    # Phase 6C-1 — retrieval-explanation fields. For a direct SKU/id lookup
    # there is no free-text query, so the score is the perfect 1.0 and the
    # reason is "direct lookup by sku/id".
    excerpt: str = ""
    exclusions_excerpt: str = ""
    match_score: float = 1.0
    match_reason: str = "direct lookup"
    matched_fields: list[str] = Field(default_factory=list)


class GetProductWarrantyTermsOutput(BaseModel):
    count: int
    terms: list[WarrantyTermsItem]


def _get_warranty_terms_impl(
    session: Session, inp: GetProductWarrantyTermsInput
) -> GetProductWarrantyTermsOutput:
    stmt = select(Product)
    lookup_field = ""
    lookup_value: Any = None
    if inp.sku is not None:
        stmt = stmt.where(Product.sku == inp.sku.strip())
        lookup_field = "sku"
        lookup_value = inp.sku.strip()
    else:
        stmt = stmt.where(Product.id == inp.product_id)
        lookup_field = "product_id"
        lookup_value = inp.product_id
    product = session.execute(stmt).scalar_one_or_none()
    if product is None:
        raise ResourceNotFoundError("product not found")

    rows = (
        session.execute(
            select(ProductWarrantyTerms)
            .where(ProductWarrantyTerms.product_id == product.id)
            .order_by(ProductWarrantyTerms.id)
        )
        .scalars()
        .all()
    )
    items = [
        WarrantyTermsItem(
            warranty_id=w.id,
            product_id=product.id,
            product_name=product.name,
            product_sku=product.sku,
            warranty_type=w.warranty_type,
            duration_months=w.duration_months,
            body=w.body,
            exclusions=w.exclusions,
            excerpt=make_excerpt(w.body),
            exclusions_excerpt=make_excerpt(w.exclusions, max_len=200),
            match_score=1.0,
            match_reason=f"direct lookup by {lookup_field}={lookup_value!r}",
            matched_fields=[lookup_field],
        )
        for w in rows
    ]
    return GetProductWarrantyTermsOutput(count=len(items), terms=items)


get_product_warranty_terms = Tool(
    name="get_product_warranty_terms",
    description=(
        "Return all warranty terms for a commerce product. Lookup by sku or "
        "product_id. Each term carries product_name, warranty_type, "
        "duration_months, full body, exclusions, and convenience excerpt + "
        "exclusions_excerpt fields plus match_score / match_reason."
    ),
    domain="commerce",
    input_schema=GetProductWarrantyTermsInput,
    output_schema=GetProductWarrantyTermsOutput,
    risk_level="low",
    read_only=True,
    impl=_get_warranty_terms_impl,
)


# ---------------------------------------------------------------------------
# 3. search_internal_agent_notes
# ---------------------------------------------------------------------------


class SearchInternalAgentNotesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: int = Field(description="Internal customer id; required.")
    query: Optional[str] = Field(
        default=None,
        description="Optional free-text match on note body and note_type.",
    )
    limit: int = Field(default=10, ge=1, le=50)


class InternalAgentNoteItem(BaseModel):
    id: int
    customer_id: int
    note_type: str
    body_excerpt: str
    related_type: Optional[str]
    related_id: Optional[int]
    created_at: datetime
    # Phase 6C-1
    excerpt: str = ""
    match_score: float = 0.0
    match_reason: str = ""
    matched_fields: list[str] = Field(default_factory=list)


class SearchInternalAgentNotesOutput(BaseModel):
    count: int
    notes: list[InternalAgentNoteItem]
    query_terms: list[str] = Field(default_factory=list)


def _search_notes_impl(
    session: Session, inp: SearchInternalAgentNotesInput
) -> SearchInternalAgentNotesOutput:
    exists = session.execute(
        select(Customer.id).where(Customer.id == inp.customer_id)
    ).scalar_one_or_none()
    if exists is None:
        raise ResourceNotFoundError("customer not found")

    terms = expand_query(inp.query) if inp.query else []
    filters = [InternalAgentNote.customer_id == inp.customer_id]
    if terms:
        filters.append(
            or_(
                _ilike_or(InternalAgentNote.body, terms),
                _ilike_or(InternalAgentNote.note_type, terms),
            )
        )

    rows = (
        session.execute(
            select(InternalAgentNote)
            .where(and_(*filters))
            .order_by(InternalAgentNote.created_at.desc())
            .limit(max(inp.limit * 3, 30) if terms else inp.limit)
        )
        .scalars()
        .all()
    )

    items: list[InternalAgentNoteItem] = []
    for n in rows:
        if terms:
            score, fields, reason = score_match(
                terms,
                {"note_type": n.note_type, "body": n.body},
            )
        else:
            score, fields, reason = (1.0, ["customer_id"], "all notes for customer")
        excerpt = make_excerpt(n.body)
        items.append(
            InternalAgentNoteItem(
                id=n.id,
                customer_id=n.customer_id,
                note_type=n.note_type,
                body_excerpt=excerpt,
                related_type=n.related_type,
                related_id=n.related_id,
                created_at=n.created_at,
                excerpt=excerpt,
                match_score=score,
                match_reason=reason,
                matched_fields=fields,
            )
        )
    if terms:
        items.sort(key=lambda x: (-x.match_score, x.created_at), reverse=False)
        # The lambda above sorts ASC by created_at within equal scores; flip
        # to newest-first explicitly.
        items.sort(key=lambda x: (-x.match_score, -x.created_at.timestamp()))
    items = items[: inp.limit]
    return SearchInternalAgentNotesOutput(
        count=len(items), notes=items, query_terms=terms
    )


search_internal_agent_notes = Tool(
    name="search_internal_agent_notes",
    description=(
        "Search internal agent notes for a customer, optionally narrowed by "
        "free-text (matched against note body and note_type with synonym "
        "expansion). Newest first; up to 50 notes. Each row carries "
        "match_score / match_reason / matched_fields / excerpt."
    ),
    domain="support",
    input_schema=SearchInternalAgentNotesInput,
    output_schema=SearchInternalAgentNotesOutput,
    risk_level="medium",
    read_only=True,
    impl=_search_notes_impl,
)


# ---------------------------------------------------------------------------
# 4. search_operational_incidents
# ---------------------------------------------------------------------------


class SearchOperationalIncidentsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: Optional[str] = Field(
        default=None, description="Optional free-text match on title and body."
    )
    domain: Optional[str] = Field(default=None)
    active_only: bool = Field(
        default=False, description="If true, only incidents without resolved_at."
    )
    limit: int = Field(default=10, ge=1, le=50)

    @model_validator(mode="after")
    def _at_least_one(self) -> "SearchOperationalIncidentsInput":
        if not any((self.query, self.domain)) and not self.active_only:
            raise ValueError(
                "provide at least one of query, domain, or active_only=True"
            )
        return self


class IncidentItem(BaseModel):
    id: int
    domain: str
    incident_type: str
    title: str
    body_excerpt: str
    started_at: datetime
    resolved_at: Optional[datetime]
    affected_entities_json: Optional[dict[str, Any]]
    # Phase 6C-1
    excerpt: str = ""
    match_score: float = 0.0
    match_reason: str = ""
    matched_fields: list[str] = Field(default_factory=list)


class SearchOperationalIncidentsOutput(BaseModel):
    count: int
    incidents: list[IncidentItem]
    query_terms: list[str] = Field(default_factory=list)


def _search_incidents_impl(
    session: Session, inp: SearchOperationalIncidentsInput
) -> SearchOperationalIncidentsOutput:
    terms = expand_query(inp.query) if inp.query else []
    filters = []
    if terms:
        filters.append(
            or_(
                _ilike_or(OperationalIncident.title, terms),
                _ilike_or(OperationalIncident.body, terms),
                _ilike_or(OperationalIncident.incident_type, terms),
            )
        )
    if inp.domain:
        filters.append(OperationalIncident.domain == inp.domain.strip())
    if inp.active_only:
        filters.append(OperationalIncident.resolved_at.is_(None))

    rows = (
        session.execute(
            select(OperationalIncident)
            .where(and_(*filters) if filters else True)
            .order_by(OperationalIncident.started_at.desc())
            .limit(max(inp.limit * 3, 30) if terms else inp.limit)
        )
        .scalars()
        .all()
    )
    items: list[IncidentItem] = []
    for r in rows:
        if terms:
            score, fields, reason = score_match(
                terms,
                {
                    "title": r.title,
                    "incident_type": r.incident_type,
                    "body": r.body,
                },
            )
        else:
            score, fields, reason = (1.0, ["domain"], "domain/active filter only")
        excerpt = make_excerpt(r.body)
        items.append(
            IncidentItem(
                id=r.id,
                domain=r.domain,
                incident_type=r.incident_type,
                title=r.title,
                body_excerpt=excerpt,
                started_at=r.started_at,
                resolved_at=r.resolved_at,
                affected_entities_json=r.affected_entities_json,
                excerpt=excerpt,
                match_score=score,
                match_reason=reason,
                matched_fields=fields,
            )
        )
    if terms:
        items.sort(key=lambda x: (-x.match_score, -x.started_at.timestamp()))
    items = items[: inp.limit]
    return SearchOperationalIncidentsOutput(
        count=len(items), incidents=items, query_terms=terms
    )


search_operational_incidents = Tool(
    name="search_operational_incidents",
    description=(
        "Search operational incidents (outages, disruptions, delays) by "
        "domain, free-text on title/body/incident_type (with synonym "
        "expansion), and active-only state. Newest first; each row carries "
        "match_score / match_reason / matched_fields / excerpt."
    ),
    domain="support",
    input_schema=SearchOperationalIncidentsInput,
    output_schema=SearchOperationalIncidentsOutput,
    risk_level="low",
    read_only=True,
    impl=_search_incidents_impl,
)


# ---------------------------------------------------------------------------
# 5. get_support_resolution_template
# ---------------------------------------------------------------------------


class GetSupportResolutionTemplateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Optional[str] = Field(
        default=None, description="Exact category match (e.g. 'refund_delay')."
    )
    query: Optional[str] = Field(
        default=None, description="Free-text match on title and body."
    )
    limit: int = Field(default=5, ge=1, le=25)

    @model_validator(mode="after")
    def _at_least_one(self) -> "GetSupportResolutionTemplateInput":
        if not self.category and not self.query:
            raise ValueError("provide at least one of category, query")
        return self


class SupportTemplateItem(BaseModel):
    id: int
    category: str
    title: str
    body: str
    escalation_required: bool
    # Phase 6C-1
    excerpt: str = ""
    match_score: float = 0.0
    match_reason: str = ""
    matched_fields: list[str] = Field(default_factory=list)


class GetSupportResolutionTemplateOutput(BaseModel):
    count: int
    templates: list[SupportTemplateItem]
    query_terms: list[str] = Field(default_factory=list)


def _get_support_template_impl(
    session: Session, inp: GetSupportResolutionTemplateInput
) -> GetSupportResolutionTemplateOutput:
    terms = expand_query(inp.query) if inp.query else []
    filters = []
    if inp.category:
        filters.append(SupportResolutionTemplate.category == inp.category.strip())
    if terms:
        filters.append(
            or_(
                _ilike_or(SupportResolutionTemplate.title, terms),
                _ilike_or(SupportResolutionTemplate.body, terms),
                _ilike_or(SupportResolutionTemplate.category, terms),
            )
        )

    rows = (
        session.execute(
            select(SupportResolutionTemplate)
            .where(and_(*filters) if filters else True)
            .order_by(SupportResolutionTemplate.id)
            .limit(max(inp.limit * 3, 15) if terms else inp.limit)
        )
        .scalars()
        .all()
    )
    items: list[SupportTemplateItem] = []
    for t in rows:
        if terms:
            score, fields, reason = score_match(
                terms,
                {"title": t.title, "category": t.category, "body": t.body},
            )
        else:
            score, fields, reason = (1.0, ["category"], "direct category lookup")
        items.append(
            SupportTemplateItem(
                id=t.id,
                category=t.category,
                title=t.title,
                body=t.body,
                escalation_required=t.escalation_required,
                excerpt=make_excerpt(t.body),
                match_score=score,
                match_reason=reason,
                matched_fields=fields,
            )
        )
    if terms:
        items.sort(key=lambda x: (-x.match_score, x.id))
    items = items[: inp.limit]
    return GetSupportResolutionTemplateOutput(
        count=len(items), templates=items, query_terms=terms
    )


get_support_resolution_template = Tool(
    name="get_support_resolution_template",
    description=(
        "Fetch support resolution templates by category and/or free-text query "
        "(synonym-expanded against title, body, and category). Returns ranked "
        "templates with an escalation_required flag and match metadata."
    ),
    domain="support",
    input_schema=GetSupportResolutionTemplateInput,
    output_schema=GetSupportResolutionTemplateOutput,
    risk_level="low",
    read_only=True,
    impl=_get_support_template_impl,
)


# ---------------------------------------------------------------------------
# 6. (BONUS) list_policy_versions
# ---------------------------------------------------------------------------


class ListPolicyVersionsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str = Field(description="Domain (airline/commerce/saas/support/crm).")
    policy_type: str = Field(
        description="Policy type (refund_policy, baggage_policy, etc.)."
    )


class PolicyVersionItem(BaseModel):
    id: int
    title: str
    version: int
    effective_from: _date
    effective_to: Optional[_date]
    is_active: bool


class ListPolicyVersionsOutput(BaseModel):
    domain: str
    policy_type: str
    count: int
    versions: list[PolicyVersionItem]


def _list_versions_impl(
    session: Session, inp: ListPolicyVersionsInput
) -> ListPolicyVersionsOutput:
    rows = (
        session.execute(
            select(PolicyDocument)
            .where(
                PolicyDocument.domain == inp.domain.strip(),
                PolicyDocument.policy_type == inp.policy_type.strip(),
            )
            .order_by(PolicyDocument.version.desc(), PolicyDocument.id.desc())
        )
        .scalars()
        .all()
    )
    if not rows:
        raise ResourceNotFoundError(
            f"no policy for domain={inp.domain!r}, policy_type={inp.policy_type!r}"
        )
    items = [
        PolicyVersionItem(
            id=p.id,
            title=p.title,
            version=p.version,
            effective_from=p.effective_from,
            effective_to=p.effective_to,
            is_active=p.is_active,
        )
        for p in rows
    ]
    return ListPolicyVersionsOutput(
        domain=inp.domain.strip(),
        policy_type=inp.policy_type.strip(),
        count=len(items),
        versions=items,
    )


list_policy_versions = Tool(
    name="list_policy_versions",
    description=(
        "List every version (current + historical) of a (domain, policy_type) "
        "policy. Use this to audit policy changes over time or to find an "
        "earlier version that was in effect on a given date."
    ),
    domain="kb",
    input_schema=ListPolicyVersionsInput,
    output_schema=ListPolicyVersionsOutput,
    risk_level="low",
    read_only=True,
    impl=_list_versions_impl,
)


# ---------------------------------------------------------------------------
# 7. (BONUS) get_active_policy
# ---------------------------------------------------------------------------


class GetActivePolicyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str = Field(description="Domain (airline/commerce/saas/support/crm).")
    policy_type: str = Field(description="Policy type (refund_policy, etc.).")


class GetActivePolicyOutput(BaseModel):
    id: int
    domain: str
    policy_type: str
    title: str
    version: int
    effective_from: _date
    effective_to: Optional[_date]
    body: str
    is_active: bool


def _get_active_policy_impl(
    session: Session, inp: GetActivePolicyInput
) -> GetActivePolicyOutput:
    row = (
        session.execute(
            select(PolicyDocument)
            .where(
                PolicyDocument.domain == inp.domain.strip(),
                PolicyDocument.policy_type == inp.policy_type.strip(),
                PolicyDocument.is_active.is_(True),
            )
            .order_by(PolicyDocument.version.desc())
            .limit(1)
        )
        .scalar_one_or_none()
    )
    if row is None:
        raise ResourceNotFoundError(
            f"no active policy for domain={inp.domain!r}, policy_type={inp.policy_type!r}"
        )
    return GetActivePolicyOutput(
        id=row.id,
        domain=row.domain,
        policy_type=row.policy_type,
        title=row.title,
        version=row.version,
        effective_from=row.effective_from,
        effective_to=row.effective_to,
        body=row.body,
        is_active=row.is_active,
    )


get_active_policy = Tool(
    name="get_active_policy",
    description=(
        "Return the currently-active policy document (highest active version) "
        "for a given domain + policy_type combination. Returns the full body, "
        "not just an excerpt."
    ),
    domain="kb",
    input_schema=GetActivePolicyInput,
    output_schema=GetActivePolicyOutput,
    risk_level="low",
    read_only=True,
    impl=_get_active_policy_impl,
)


# Re-export the infer_policy_types helper so the rewritten policy.py /
# policy_extras.py tools can use it without circular-import gymnastics.
__all__ = [
    "search_return_rules",
    "get_product_warranty_terms",
    "search_internal_agent_notes",
    "search_operational_incidents",
    "get_support_resolution_template",
    "list_policy_versions",
    "get_active_policy",
    "infer_policy_types",
]
