"""Commerce product tools (Phase C1): search_products, get_product_details, check_product_inventory."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session, aliased, selectinload

from app.models import (
    Product,
    ProductAttribute,
    ProductCategory,
    ProductInventory,
    ProductPrice,
    Warehouse,
)
from app.tools.base import ResourceNotFoundError, Tool


# ---------------------------------------------------------------------------
# search_products
# ---------------------------------------------------------------------------


class SearchProductsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2)
    category: Optional[str] = Field(
        default=None, description="Partial match against the category name."
    )
    max_price: Optional[Decimal] = Field(default=None, gt=0)
    limit: int = Field(default=10, ge=1, le=50)


class ProductSearchItem(BaseModel):
    product_id: int
    sku: str
    name: str
    brand: Optional[str]
    category_name: str
    current_price: Optional[Decimal]
    currency: Optional[str]


class SearchProductsOutput(BaseModel):
    count: int
    products: list[ProductSearchItem]


def _search_products_impl(
    session: Session, inp: SearchProductsInput
) -> SearchProductsOutput:
    pattern = f"%{inp.query.strip()}%"
    current_price = aliased(ProductPrice)

    stmt = (
        select(
            Product.id,
            Product.sku,
            Product.name,
            Product.brand,
            ProductCategory.name.label("category_name"),
            current_price.price,
            current_price.currency,
        )
        .join(ProductCategory, Product.category_id == ProductCategory.id)
        .outerjoin(
            current_price,
            and_(
                current_price.product_id == Product.id,
                current_price.valid_to.is_(None),
            ),
        )
        .where(Product.is_active.is_(True))
        .where(Product.name.ilike(pattern))
    )
    if inp.category is not None:
        stmt = stmt.where(ProductCategory.name.ilike(f"%{inp.category.strip()}%"))
    if inp.max_price is not None:
        stmt = stmt.where(current_price.price <= inp.max_price)
    stmt = stmt.order_by(Product.name).limit(inp.limit)

    rows = session.execute(stmt).all()
    items = [
        ProductSearchItem(
            product_id=pid,
            sku=sku,
            name=name,
            brand=brand,
            category_name=cat,
            current_price=price,
            currency=currency,
        )
        for pid, sku, name, brand, cat, price, currency in rows
    ]
    return SearchProductsOutput(count=len(items), products=items)


search_products = Tool(
    name="search_products",
    description=(
        "Search active commerce products by free-text query (matched against "
        "name). Optionally filter by category name (partial match) and a "
        "max_price ceiling. Returns up to 50 active products with current price."
    ),
    domain="commerce",
    input_schema=SearchProductsInput,
    output_schema=SearchProductsOutput,
    risk_level="low",
    read_only=True,
    impl=_search_products_impl,
)


# ---------------------------------------------------------------------------
# get_product_details
# ---------------------------------------------------------------------------


class GetProductDetailsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku: Optional[str] = Field(default=None)
    product_id: Optional[int] = Field(default=None)

    @model_validator(mode="after")
    def _exactly_one(self) -> "GetProductDetailsInput":
        provided = sum(x is not None for x in (self.sku, self.product_id))
        if provided != 1:
            raise ValueError("provide exactly one of sku, product_id")
        return self


class ProductAttributeItem(BaseModel):
    name: str
    value: str


class GetProductDetailsOutput(BaseModel):
    product_id: int
    sku: str
    name: str
    brand: Optional[str]
    category_name: str
    description: Optional[str]
    is_active: bool
    current_price: Optional[Decimal]
    currency: Optional[str]
    attributes: list[ProductAttributeItem]
    total_inventory: int


def _product_details_impl(
    session: Session, inp: GetProductDetailsInput
) -> GetProductDetailsOutput:
    stmt = select(Product).options(
        selectinload(Product.attributes),
        selectinload(Product.category),
    )
    if inp.sku is not None:
        stmt = stmt.where(Product.sku == inp.sku.strip())
    else:
        stmt = stmt.where(Product.id == inp.product_id)
    product = session.execute(stmt).scalar_one_or_none()
    if product is None:
        raise ResourceNotFoundError("product not found")

    current = session.execute(
        select(ProductPrice)
        .where(ProductPrice.product_id == product.id, ProductPrice.valid_to.is_(None))
        .order_by(ProductPrice.id.desc())
        .limit(1)
    ).scalar_one_or_none()
    inventory_total = session.execute(
        select(func.coalesce(func.sum(ProductInventory.quantity_available), 0))
        .where(ProductInventory.product_id == product.id)
    ).scalar_one()

    return GetProductDetailsOutput(
        product_id=product.id,
        sku=product.sku,
        name=product.name,
        brand=product.brand,
        category_name=product.category.name,
        description=product.description,
        is_active=product.is_active,
        current_price=(current.price if current else None),
        currency=(current.currency if current else None),
        attributes=[
            ProductAttributeItem(name=a.attribute_name, value=a.attribute_value)
            for a in product.attributes
        ],
        total_inventory=int(inventory_total or 0),
    )


get_product_details = Tool(
    name="get_product_details",
    description=(
        "Return full details for a commerce product (sku, name, brand, "
        "category, current price, attributes, and total inventory across "
        "warehouses). Lookup by sku or product_id."
    ),
    domain="commerce",
    input_schema=GetProductDetailsInput,
    output_schema=GetProductDetailsOutput,
    risk_level="low",
    read_only=True,
    impl=_product_details_impl,
)


# ---------------------------------------------------------------------------
# check_product_inventory
# ---------------------------------------------------------------------------


class CheckProductInventoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku: Optional[str] = Field(default=None)
    product_id: Optional[int] = Field(default=None)
    city: Optional[str] = Field(
        default=None, description="Optional partial match against the warehouse city."
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> "CheckProductInventoryInput":
        provided = sum(x is not None for x in (self.sku, self.product_id))
        if provided != 1:
            raise ValueError("provide exactly one of sku, product_id")
        return self


class InventoryItem(BaseModel):
    warehouse_id: int
    warehouse_name: str
    city: str
    country: str
    quantity_available: int
    updated_at: datetime


class CheckProductInventoryOutput(BaseModel):
    product_id: int
    sku: str
    name: str
    city_filter: Optional[str]
    warehouse_count: int
    total_quantity: int
    inventory: list[InventoryItem]


def _inventory_impl(
    session: Session, inp: CheckProductInventoryInput
) -> CheckProductInventoryOutput:
    stmt = select(Product)
    if inp.sku is not None:
        stmt = stmt.where(Product.sku == inp.sku.strip())
    else:
        stmt = stmt.where(Product.id == inp.product_id)
    product = session.execute(stmt).scalar_one_or_none()
    if product is None:
        raise ResourceNotFoundError("product not found")

    inv_stmt = (
        select(
            Warehouse.id,
            Warehouse.name,
            Warehouse.city,
            Warehouse.country,
            ProductInventory.quantity_available,
            ProductInventory.updated_at,
        )
        .join(Warehouse, ProductInventory.warehouse_id == Warehouse.id)
        .where(ProductInventory.product_id == product.id)
        .order_by(Warehouse.name)
    )
    if inp.city is not None:
        inv_stmt = inv_stmt.where(Warehouse.city.ilike(f"%{inp.city.strip()}%"))

    rows = session.execute(inv_stmt).all()
    items = [
        InventoryItem(
            warehouse_id=wid,
            warehouse_name=wname,
            city=city,
            country=country,
            quantity_available=qty,
            updated_at=updated_at,
        )
        for wid, wname, city, country, qty, updated_at in rows
    ]
    return CheckProductInventoryOutput(
        product_id=product.id,
        sku=product.sku,
        name=product.name,
        city_filter=inp.city,
        warehouse_count=len(items),
        total_quantity=sum(i.quantity_available for i in items),
        inventory=items,
    )


check_product_inventory = Tool(
    name="check_product_inventory",
    description=(
        "Return per-warehouse inventory for a product, optionally filtered to "
        "warehouses in cities matching a substring. Lookup by sku or product_id."
    ),
    domain="commerce",
    input_schema=CheckProductInventoryInput,
    output_schema=CheckProductInventoryOutput,
    risk_level="low",
    read_only=True,
    impl=_inventory_impl,
)
