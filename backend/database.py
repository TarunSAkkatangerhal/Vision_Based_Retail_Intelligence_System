import logging
import sqlite3
from typing import Any, Dict, List, Optional

from shared.config import DATABASE_PATH

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Database initialisation
# ──────────────────────────────────────────────

def _get_connection() -> sqlite3.Connection:
    """Return a new connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create all required tables if they do not already exist."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inventory_logs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            schema_version  TEXT DEFAULT '1.0',
            timestamp       TEXT NOT NULL,
            camera_id       TEXT NOT NULL,
            product_name    TEXT NOT NULL,
            shelf_id        TEXT NOT NULL,
            count           INTEGER NOT NULL,
            status          TEXT NOT NULL,
            created_at      TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customer_logs (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            schema_version      TEXT DEFAULT '1.0',
            timestamp           TEXT NOT NULL,
            camera_id           TEXT NOT NULL,
            customer_id         TEXT NOT NULL,
            shelf_id            TEXT NOT NULL,
            dwell_time_seconds  INTEGER NOT NULL,
            interaction         TEXT NOT NULL,
            item_image_path     TEXT DEFAULT NULL,
            created_at          TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Migration: add item_image_path if missing (existing databases)
    try:
        cursor.execute("ALTER TABLE customer_logs ADD COLUMN item_image_path TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        pass  # column already exists

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS promotions (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            shelf_id          TEXT NOT NULL,
            reason            TEXT NOT NULL,
            confidence_score  REAL NOT NULL,
            suggested_action  TEXT NOT NULL,
            generated_at      TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── Indexes for query performance ──
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_inv_shelf_ts
        ON inventory_logs(shelf_id, timestamp DESC)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_cust_shelf_ts
        ON customer_logs(shelf_id, timestamp DESC)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_cust_timestamp
        ON customer_logs(timestamp DESC)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_inv_timestamp
        ON inventory_logs(timestamp DESC)
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialised at %s", DATABASE_PATH)


# ──────────────────────────────────────────────
# Insert helpers
# ──────────────────────────────────────────────

def insert_inventory(data: Dict[str, Any]) -> int:
    """
    Insert inventory records from a validated Inventory Schema dict.

    Returns:
        Number of rows inserted.
    """
    conn = _get_connection()
    cursor = conn.cursor()
    rows_inserted = 0

    schema_version = data.get("schema_version", "1.0")
    timestamp = data["timestamp"]
    camera_id = data["camera_id"]

    for product in data["products"]:
        cursor.execute(
            """
            INSERT INTO inventory_logs
                (schema_version, timestamp, camera_id, product_name, shelf_id, count, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                schema_version,
                timestamp,
                camera_id,
                product["product_name"],
                product["shelf_id"],
                product["count"],
                product["status"],
            ),
        )
        rows_inserted += 1

    conn.commit()
    conn.close()
    logger.info("Inserted %d inventory records.", rows_inserted)
    return rows_inserted


def insert_customers(data: Dict[str, Any]) -> int:
    """
    Insert customer behaviour records from a validated Customer Behavior Schema dict.

    Returns:
        Number of rows inserted.
    """
    conn = _get_connection()
    cursor = conn.cursor()
    rows_inserted = 0

    schema_version = data.get("schema_version", "1.0")
    timestamp = data["timestamp"]
    camera_id = data["camera_id"]

    for customer in data["customers"]:
        cursor.execute(
            """
            INSERT INTO customer_logs
                (schema_version, timestamp, camera_id, customer_id, shelf_id,
                 dwell_time_seconds, interaction, item_image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                schema_version,
                timestamp,
                camera_id,
                customer["customer_id"],
                customer["shelf_id"],
                customer["dwell_time_seconds"],
                customer["interaction"],
                customer.get("item_image_path"),
            ),
        )
        rows_inserted += 1

    conn.commit()
    conn.close()
    logger.info("Inserted %d customer records.", rows_inserted)
    return rows_inserted


# ──────────────────────────────────────────────
# Query helpers
# ──────────────────────────────────────────────

def get_promotions() -> List[Dict[str, Any]]:
    """Return all rows from the promotions table as a list of dicts."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT shelf_id, reason, confidence_score, suggested_action, generated_at "
        "FROM promotions ORDER BY generated_at DESC"
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def insert_promotions(promotions: List[Dict[str, Any]]) -> int:
    """
    Insert promotion records into the promotions table.

    Returns:
        Number of rows inserted.
    """
    conn = _get_connection()
    cursor = conn.cursor()
    rows_inserted = 0

    for promo in promotions:
        cursor.execute(
            """
            INSERT INTO promotions
                (shelf_id, reason, confidence_score, suggested_action, generated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                promo["shelf_id"],
                promo["reason"],
                promo["confidence_score"],
                promo["suggested_action"],
                promo.get("generated_at"),
            ),
        )
        rows_inserted += 1

    conn.commit()
    conn.close()
    logger.info("Inserted %d promotion records.", rows_inserted)
    return rows_inserted


def get_inventory_logs(shelf_id: str | None = None, limit: int = 100) -> List[Dict[str, Any]]:
    """Return recent inventory log entries, optionally filtered by shelf."""
    conn = _get_connection()
    cursor = conn.cursor()
    if shelf_id:
        cursor.execute(
            "SELECT * FROM inventory_logs WHERE shelf_id = ? ORDER BY timestamp DESC LIMIT ?",
            (shelf_id, limit),
        )
    else:
        cursor.execute(
            "SELECT * FROM inventory_logs ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_customer_logs(shelf_id: str | None = None, limit: int = 100) -> List[Dict[str, Any]]:
    """Return recent customer behaviour log entries, optionally filtered by shelf."""
    conn = _get_connection()
    cursor = conn.cursor()
    if shelf_id:
        cursor.execute(
            "SELECT * FROM customer_logs WHERE shelf_id = ? ORDER BY timestamp DESC LIMIT ?",
            (shelf_id, limit),
        )
    else:
        cursor.execute(
            "SELECT * FROM customer_logs ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def check_health() -> bool:
    """Return True if the database is reachable and the tables exist."""
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row["name"] for row in cursor.fetchall()}
        conn.close()
        required = {"inventory_logs", "customer_logs", "promotions"}
        return required.issubset(tables)
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return False
