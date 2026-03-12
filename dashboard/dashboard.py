import logging
import time

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from shared.config import API_BASE_URL

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Retail Intelligence Dashboard",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 Retail Intelligence Dashboard")


# ──────────────────────────────────────────────
# API helpers
# ──────────────────────────────────────────────

def fetch_health() -> dict | None:
    """GET /health from the backend API."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.error("Failed to fetch /health: %s", e)
        return None


def fetch_promotions() -> dict | None:
    """GET /promotions from the backend API."""
    try:
        resp = requests.get(f"{API_BASE_URL}/promotions", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.error("Failed to fetch /promotions: %s", e)
        return None


def fetch_inventory_logs() -> list | None:
    """GET /inventory/logs if available, else return None."""
    try:
        resp = requests.get(f"{API_BASE_URL}/inventory/logs", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def fetch_customer_logs() -> list | None:
    """GET /customers/logs if available, else return None."""
    try:
        resp = requests.get(f"{API_BASE_URL}/customers/logs", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


# ──────────────────────────────────────────────
# Health Status
# ──────────────────────────────────────────────

st.header("System Health")
health = fetch_health()

if health:
    col1, col2 = st.columns(2)
    col1.metric("API Status", health.get("status", "unknown"))
    col2.metric("Database Connected", "Yes" if health.get("db_connected") else "No")
else:
    st.error("⚠️ Cannot reach backend API. Is the server running?")


# ──────────────────────────────────────────────
# Inventory Status Table
# ──────────────────────────────────────────────

st.header("📦 Inventory Status")
inv_logs = fetch_inventory_logs()

if inv_logs:
    inv_df = pd.DataFrame(inv_logs)
    # Show latest status per product/shelf
    if not inv_df.empty:
        latest = inv_df.sort_values("timestamp").groupby(["shelf_id", "product_name"]).last().reset_index()
        st.dataframe(
            latest[["shelf_id", "product_name", "count", "status", "timestamp"]],
            use_container_width=True,
        )
    else:
        st.info("No inventory data yet.")
else:
    st.info("No inventory data available (inventory/logs endpoint not reachable).")


# ──────────────────────────────────────────────
# Customer Dwell Heatmap
# ──────────────────────────────────────────────

st.header("🧍 Customer Dwell Heatmap")
cust_logs = fetch_customer_logs()

if cust_logs:
    cust_df = pd.DataFrame(cust_logs)
    if not cust_df.empty and "shelf_id" in cust_df.columns and "dwell_time_seconds" in cust_df.columns:
        pivot = cust_df.groupby(["shelf_id", "customer_id"])["dwell_time_seconds"].sum().reset_index()
        heatmap_data = pivot.pivot_table(
            index="shelf_id",
            columns="customer_id",
            values="dwell_time_seconds",
            fill_value=0,
        )
        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Customer", y="Shelf", color="Dwell (s)"),
            title="Dwell Time by Shelf and Customer",
            color_continuous_scale="YlOrRd",
            aspect="auto",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No customer dwell data yet.")
else:
    st.info("No customer data available (customers/logs endpoint not reachable).")


# ──────────────────────────────────────────────
# Promotion Suggestions
# ──────────────────────────────────────────────

st.header("🎯 Promotion Suggestions")
promo_data = fetch_promotions()

if promo_data and promo_data.get("promotions"):
    promo_df = pd.DataFrame(promo_data["promotions"])
    st.dataframe(
        promo_df[["shelf_id", "reason", "confidence_score", "suggested_action"]],
        use_container_width=True,
    )
else:
    st.info("No promotion suggestions generated yet.")


# ──────────────────────────────────────────────
# Auto-refresh every 30 seconds
# ──────────────────────────────────────────────

st.caption("Dashboard auto-refreshes every 30 seconds.")
time.sleep(30)
st.rerun()
