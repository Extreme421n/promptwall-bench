"""Tool registry and 15 read-only tools backed by the seeded DB.

Phase 2E adds 7 tools that deliberately overlap with the original 8, so the
chatbot has to pick the most specific one rather than defaulting to the
broadest. The overlap is realistic: an airline agent's toolbox has multiple
ways to answer "what's my loyalty balance?" or "what's the cancellation
policy?", and the right answer depends on context.
"""

from app.tools.base import (
    AmbiguousInputError,
    InvocationResult,
    ResourceNotFoundError,
    Tool,
    ToolError,
    ToolNotFoundError,
    ToolValidationError,
    default_registry,
    invoke_tool,
)
from app.tools.baggage import get_baggage_policy
from app.tools.booking import get_booking_details
from app.tools.change import calculate_change_fee, search_change_options
from app.tools.commerce_extras import (
    calculate_bundle_price,
    get_commerce_return_status,
)
from app.tools.commerce_orders import (
    get_commerce_order_status,
    get_commerce_refund_status,
    get_shipment_status,
)
from app.tools.commerce_products import (
    check_product_inventory,
    get_product_details,
    search_products,
)
from app.tools.customer import get_customer_profile
from app.tools.customer_search import search_customer_records
from app.tools.customer_segment import get_customer_segment
from app.tools.flight import get_flight_status, search_available_flights
from app.tools.issue import get_customer_open_issues
from app.tools.kb import search_kb_articles
from app.tools.loyalty import get_loyalty_balance
from app.tools.policy import get_policy_clause
from app.tools.policy_extras import (
    get_latest_policy_version,
    search_policy_documents,
)
from app.tools.refund import get_refund_status
from app.tools.saas_invoice import get_invoice_status
from app.tools.saas_seat_alloc import get_saas_seat_allocation
from app.tools.saas_subscription import (
    get_plan_limits,
    get_subscription_status,
)
from app.tools.saas_usage import calculate_usage_overage, get_api_usage_summary
from app.tools.seat import search_available_seats
from app.tools.support import get_support_ticket_status
from app.tools.support_extras import (
    create_support_ticket_draft,
    get_escalation_policy,
    search_support_tickets,
)
from app.tools.text_retrieval import (
    get_active_policy,
    get_product_warranty_terms,
    get_support_resolution_template,
    list_policy_versions,
    search_internal_agent_notes,
    search_operational_incidents,
    search_return_rules,
)

for _tool in (
    # Phase 1D — the original 8
    get_customer_profile,
    get_booking_details,
    get_flight_status,
    search_available_flights,
    get_refund_status,
    get_baggage_policy,
    get_support_ticket_status,
    search_kb_articles,
    # Phase 2E — 7 overlapping tools
    search_available_seats,
    calculate_change_fee,
    search_change_options,
    get_loyalty_balance,
    get_policy_clause,
    get_customer_open_issues,
    search_customer_records,
    # Phase C1 — SaaS / billing tools
    get_subscription_status,
    get_plan_limits,
    get_invoice_status,
    calculate_usage_overage,
    get_api_usage_summary,
    get_saas_seat_allocation,
    # Phase C1 — Commerce tools
    search_products,
    get_product_details,
    check_product_inventory,
    get_commerce_order_status,
    get_commerce_refund_status,
    get_shipment_status,
    # Phase C2 — Support extras
    search_support_tickets,
    get_escalation_policy,
    create_support_ticket_draft,
    # Phase C2 — KB / policy extras
    search_policy_documents,
    get_latest_policy_version,
    # Phase C2 — Commerce extras
    calculate_bundle_price,
    get_commerce_return_status,
    # Phase C2 — CRM extras
    get_customer_segment,
    # Phase 6B-4 — textual retrieval tools
    search_return_rules,
    get_product_warranty_terms,
    search_internal_agent_notes,
    search_operational_incidents,
    get_support_resolution_template,
    list_policy_versions,
    get_active_policy,
):
    default_registry.register(_tool)


__all__ = [
    "Tool",
    "InvocationResult",
    "ToolError",
    "ToolValidationError",
    "ToolNotFoundError",
    "ResourceNotFoundError",
    "AmbiguousInputError",
    "default_registry",
    "invoke_tool",
    # Phase 1D
    "get_customer_profile",
    "get_booking_details",
    "get_flight_status",
    "search_available_flights",
    "get_refund_status",
    "get_baggage_policy",
    "get_support_ticket_status",
    "search_kb_articles",
    # Phase 2E
    "search_available_seats",
    "calculate_change_fee",
    "search_change_options",
    "get_loyalty_balance",
    "get_policy_clause",
    "get_customer_open_issues",
    "search_customer_records",
    # Phase C1 — SaaS / billing
    "get_subscription_status",
    "get_plan_limits",
    "get_invoice_status",
    "calculate_usage_overage",
    "get_api_usage_summary",
    "get_saas_seat_allocation",
    # Phase C1 — Commerce
    "search_products",
    "get_product_details",
    "check_product_inventory",
    "get_commerce_order_status",
    "get_commerce_refund_status",
    "get_shipment_status",
    # Phase C2 — Support extras
    "search_support_tickets",
    "get_escalation_policy",
    "create_support_ticket_draft",
    # Phase C2 — KB / policy extras
    "search_policy_documents",
    "get_latest_policy_version",
    # Phase C2 — Commerce extras
    "calculate_bundle_price",
    "get_commerce_return_status",
    # Phase C2 — CRM extras
    "get_customer_segment",
    # Phase 6B-4 — textual retrieval
    "search_return_rules",
    "get_product_warranty_terms",
    "search_internal_agent_notes",
    "search_operational_incidents",
    "get_support_resolution_template",
    "list_policy_versions",
    "get_active_policy",
]
