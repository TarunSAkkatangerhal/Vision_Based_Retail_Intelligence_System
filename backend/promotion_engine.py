import logging
import sqlite3
from typing import Any, Dict, List

import pandas as pd

from shared.config import DATABASE_PATH, DWELL_TIME_THRESHOLD, LOW_STOCK_THRESHOLD
from shared.schemas import PromotionData, PromotionItem
from shared.utils import get_current_timestamp
from backend.database import insert_promotions

logger = logging.getLogger(__name__)


def _get_connection() -> sqlite3.Connection:
    """Return a new connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ──────────────────────────────────────────────
# Rule 1 — High dwell time + low conversion
# ──────────────────────────────────────────────

def _detect_high_dwell_low_conversion() -> List[Dict[str, Any]]:
    """
    Find shelves where customers spend a long time but rarely pick up products.

    Logic:
      • Average dwell_time_seconds > DWELL_TIME_THRESHOLD
      • Ratio of 'picked_product' interactions to total visits < 0.3
    """
    conn = _get_connection()
    df = pd.read_sql_query("SELECT * FROM customer_logs", conn)
    conn.close()

    if df.empty:
        return []

    results: List[Dict[str, Any]] = []
    for shelf_id, group in df.groupby("shelf_id"):
        avg_dwell = group["dwell_time_seconds"].mean()
        total = len(group)
        picks = len(group[group["interaction"] == "picked_product"])
        conversion_rate = picks / total if total > 0 else 0.0

        if avg_dwell > DWELL_TIME_THRESHOLD and conversion_rate < 0.3:
            confidence = min(1.0, round(avg_dwell / (DWELL_TIME_THRESHOLD * 3), 2))
            results.append({
                "shelf_id": shelf_id,
                "reason": "high_dwell_low_conversion",
                "confidence_score": confidence,
                "suggested_action": (
                    f"Customers browse shelf {shelf_id} for avg {avg_dwell:.0f}s "
                    f"but only {conversion_rate:.0%} pick a product. "
                    "Consider better signage or product placement."
                ),
            })

    return results


# ──────────────────────────────────────────────
# Rule 2 — Low stock + high interest
# ──────────────────────────────────────────────

def _detect_low_stock_high_interest() -> List[Dict[str, Any]]:
    """
    Find shelves that have low stock while customers keep visiting them.

    Logic:
      • Latest inventory status for a shelf is 'low_stock' or 'empty'
      • The shelf appears frequently in customer_logs
    """
    conn = _get_connection()
    inv_df = pd.read_sql_query("SELECT * FROM inventory_logs", conn)
    cust_df = pd.read_sql_query("SELECT * FROM customer_logs", conn)
    conn.close()

    if inv_df.empty:
        return []

    results: List[Dict[str, Any]] = []

    # Get latest status per shelf
    latest_inv = inv_df.sort_values("timestamp").groupby("shelf_id").last().reset_index()
    low_shelves = latest_inv[latest_inv["status"].isin(["low_stock", "empty"])]

    if low_shelves.empty:
        return []

    # Count customer visits per shelf
    visit_counts = cust_df.groupby("shelf_id").size() if not cust_df.empty else pd.Series(dtype=int)

    for _, row in low_shelves.iterrows():
        shelf_id = row["shelf_id"]
        visits = visit_counts.get(shelf_id, 0)
        if visits >= 2:
            confidence = min(1.0, round(visits / 10, 2))
            results.append({
                "shelf_id": shelf_id,
                "reason": "low_stock_high_interest",
                "confidence_score": confidence,
                "suggested_action": (
                    f"Shelf {shelf_id} is {row['status']} but had {visits} customer visits. "
                    "Restock urgently to avoid lost sales."
                ),
            })

    return results


# ──────────────────────────────────────────────
# Rule 3 — Frequent replacements
# ──────────────────────────────────────────────

def _detect_frequent_replacements() -> List[Dict[str, Any]]:
    """
    Find shelves where customers frequently replace products.

    Logic:
      • Ratio of 'replaced_product' interactions to total visits > 0.3
    """
    conn = _get_connection()
    df = pd.read_sql_query("SELECT * FROM customer_logs", conn)
    conn.close()

    if df.empty:
        return []

    results: List[Dict[str, Any]] = []
    for shelf_id, group in df.groupby("shelf_id"):
        total = len(group)
        replacements = len(group[group["interaction"] == "replaced_product"])
        replace_rate = replacements / total if total > 0 else 0.0

        if replace_rate > 0.3 and replacements >= 2:
            confidence = min(1.0, round(replace_rate, 2))
            results.append({
                "shelf_id": shelf_id,
                "reason": "frequent_replacement",
                "confidence_score": confidence,
                "suggested_action": (
                    f"Shelf {shelf_id} has a {replace_rate:.0%} replacement rate "
                    f"({replacements}/{total} interactions). "
                    "Review pricing, packaging, or product quality."
                ),
            })

    return results


# ──────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────

def generate_promotions() -> Dict[str, Any]:
    """
    Analyse inventory_logs and customer_logs, generate promotion suggestions,
    persist them to the promotions table, and return a Promotion Schema dict.
    """
    logger.info("Running promotion engine…")

    all_promos: List[Dict[str, Any]] = []
    all_promos.extend(_detect_high_dwell_low_conversion())
    all_promos.extend(_detect_low_stock_high_interest())
    all_promos.extend(_detect_frequent_replacements())

    # De-duplicate by (shelf_id, reason)
    seen = set()
    unique_promos: List[Dict[str, Any]] = []
    for p in all_promos:
        key = (p["shelf_id"], p["reason"])
        if key not in seen:
            seen.add(key)
            unique_promos.append(p)

    generated_at = get_current_timestamp()

    # Persist to database
    for p in unique_promos:
        p["generated_at"] = generated_at
    if unique_promos:
        insert_promotions(unique_promos)

    # Build and validate Pydantic model
    promo_items = [PromotionItem(**p) for p in unique_promos]
    promo_data = PromotionData(
        schema_version="1.0",
        generated_at=generated_at,
        promotions=promo_items,
    )

    validated = PromotionData.model_validate(promo_data.model_dump())
    logger.info("Promotion engine produced %d suggestions.", len(validated.promotions))
    return validated.model_dump()
