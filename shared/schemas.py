from pydantic import BaseModel, Field
from typing import List, Literal
from datetime import datetime


# ──────────────────────────────────────────────
# Inventory Schema
# ──────────────────────────────────────────────

class ProductItem(BaseModel):
    product_name: str
    shelf_id: str
    count: int
    status: Literal["normal", "low_stock", "empty"]


class InventoryData(BaseModel):
    schema_version: str = "1.0"
    timestamp: str
    camera_id: str
    products: List[ProductItem]


# ──────────────────────────────────────────────
# Customer Behavior Schema
# ──────────────────────────────────────────────

class CustomerItem(BaseModel):
    customer_id: str
    shelf_id: str
    dwell_time_seconds: int
    interaction: Literal["none", "picked_product", "replaced_product"]


class CustomerBehaviorData(BaseModel):
    schema_version: str = "1.0"
    timestamp: str
    camera_id: str
    customers: List[CustomerItem]


# ──────────────────────────────────────────────
# Promotion Schema
# ──────────────────────────────────────────────

class PromotionItem(BaseModel):
    shelf_id: str
    reason: Literal[
        "high_dwell_low_conversion",
        "low_stock_high_interest",
        "frequent_replacement"
    ]
    confidence_score: float = Field(ge=0.0, le=1.0)
    suggested_action: str


class PromotionData(BaseModel):
    schema_version: str = "1.0"
    generated_at: str
    promotions: List[PromotionItem]
