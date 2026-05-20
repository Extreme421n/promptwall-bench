"""SQLAlchemy ORM models.

All models inherit from a single ``Base`` so Alembic and ``create_all`` see
every table from one place.
"""

from app.models.airline import (
    Airport,
    BaggageRule,
    Booking,
    Flight,
    Refund,
    Seat,
)
from app.models.base import Base
from app.models.commerce import (
    CommerceOrder,
    CommerceOrderItem,
    CommerceRefund,
    CommerceReturn,
    Product,
    ProductAttribute,
    ProductCategory,
    ProductInventory,
    ProductPrice,
    Shipment,
    Warehouse,
)
from app.models.crm import Customer, LoyaltyAccount
from app.models.evaluation import EvaluationResult, EvaluationRun
from app.models.kb import KBArticle
from app.models.observability import ChatSession, LLMCall, ToolInvocation, Trace
from app.models.promptwall import PromptWallCandidateDecision
from app.models.saas import (
    ApiUsageDaily,
    CustomerOrganization,
    Invoice,
    InvoiceItem,
    Organization,
    OverageCharge,
    Plan,
    SeatAllocation,
    Subscription,
    UsageEvent,
)
from app.models.support import SupportMessage, SupportTicket

__all__ = [
    "Base",
    # CRM
    "Customer",
    "LoyaltyAccount",
    # Airline
    "Airport",
    "Flight",
    "Booking",
    "Seat",
    "BaggageRule",
    "Refund",
    # Support
    "SupportTicket",
    "SupportMessage",
    # KB
    "KBArticle",
    # Observability
    "ChatSession",
    "Trace",
    "LLMCall",
    "ToolInvocation",
    # Evaluation
    "EvaluationRun",
    "EvaluationResult",
    # PromptWall
    "PromptWallCandidateDecision",
    # SaaS / billing
    "Organization",
    "CustomerOrganization",
    "Plan",
    "Subscription",
    "Invoice",
    "InvoiceItem",
    "UsageEvent",
    "ApiUsageDaily",
    "SeatAllocation",
    "OverageCharge",
    # Commerce / orders
    "ProductCategory",
    "Product",
    "ProductAttribute",
    "ProductPrice",
    "Warehouse",
    "ProductInventory",
    "CommerceOrder",
    "CommerceOrderItem",
    "Shipment",
    "CommerceReturn",
    "CommerceRefund",
]
