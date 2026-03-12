import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'shared' package can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
# API helpers (cached with 30-second TTL)
# ──────────────────────────────────────────────

@st.cache_data(ttl=30)
def fetch_health() -> dict | None:
    """GET /health from the backend API."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.error("Failed to fetch /health: %s", e)
        return None


@st.cache_data(ttl=30)
def fetch_promotions() -> dict | None:
    """GET /promotions from the backend API."""
    try:
        resp = requests.get(f"{API_BASE_URL}/promotions", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.error("Failed to fetch /promotions: %s", e)
        return None


@st.cache_data(ttl=30)
def fetch_inventory_logs() -> list | None:
    """GET /inventory/logs if available, else return None."""
    try:
        resp = requests.get(f"{API_BASE_URL}/inventory/logs", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


@st.cache_data(ttl=30)
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
# 🛍️ Purchase Tracking — Bought vs Interested
# ──────────────────────────────────────────────

st.header("🛍️ Purchase Tracking")

if cust_logs:
    cust_df = pd.DataFrame(cust_logs)
    if not cust_df.empty and "interaction" in cust_df.columns:

        # Map interactions to friendly labels
        label_map = {
            "picked_product": "✅ Bought",
            "replaced_product": "🔄 Put Back",
            "interested_no_buy": "👀 Interested (no buy)",
            "none": "🚶 Passed By",
        }
        cust_df["status"] = cust_df["interaction"].map(label_map).fillna(cust_df["interaction"])

        # ── KPI cards row ──
        total = len(cust_df)
        bought = len(cust_df[cust_df["interaction"] == "picked_product"])
        replaced = len(cust_df[cust_df["interaction"] == "replaced_product"])
        interested = len(cust_df[cust_df["interaction"] == "interested_no_buy"])
        conversion = (bought / total * 100) if total > 0 else 0

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Visits", total)
        k2.metric("✅ Bought", bought)
        k3.metric("👀 Interested (no buy)", interested)
        k4.metric("🔄 Put Back", replaced)
        k5.metric("Conversion Rate", f"{conversion:.1f}%")

        # ── Per-shelf breakdown ──
        st.subheader("Per-Shelf Breakdown")
        shelf_filter = st.selectbox(
            "Filter by shelf", ["All Shelves"] + sorted(cust_df["shelf_id"].unique().tolist())
        )
        if shelf_filter != "All Shelves":
            filtered = cust_df[cust_df["shelf_id"] == shelf_filter]
        else:
            filtered = cust_df

        # Interaction pie chart + table side by side
        col_chart, col_table = st.columns([1, 1])

        with col_chart:
            interaction_counts = filtered["status"].value_counts().reset_index()
            interaction_counts.columns = ["Status", "Count"]
            fig_pie = px.pie(
                interaction_counts,
                names="Status",
                values="Count",
                title="Customer Interactions",
                color="Status",
                color_discrete_map={
                    "✅ Bought": "#2ecc71",
                    "🔄 Put Back": "#f39c12",
                    "👀 Interested (no buy)": "#e74c3c",
                    "🚶 Passed By": "#3498db",
                },
            )
            fig_pie.update_traces(textinfo="value+percent")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_table:
            # Per-shelf summary table
            summary = filtered.groupby("shelf_id").agg(
                visits=("interaction", "count"),
                bought=("interaction", lambda x: (x == "picked_product").sum()),
                put_back=("interaction", lambda x: (x == "replaced_product").sum()),
                interested=("interaction", lambda x: (x == "interested_no_buy").sum()),
                avg_dwell=("dwell_time_seconds", "mean"),
            ).reset_index()
            summary["conversion"] = (summary["bought"] / summary["visits"] * 100).round(1)
            summary["avg_dwell"] = summary["avg_dwell"].round(1)
            summary.columns = ["Shelf", "Visits", "Bought", "Put Back", "Interested", "Avg Dwell (s)", "Conversion %"]
            st.dataframe(summary, use_container_width=True, hide_index=True)

        # ── Conversion over time ──
        st.subheader("Conversion Over Time")
        if "timestamp" in filtered.columns:
            time_df = filtered.copy()
            time_df["timestamp"] = pd.to_datetime(time_df["timestamp"], errors="coerce")
            time_df = time_df.dropna(subset=["timestamp"])
            if not time_df.empty:
                time_df["minute"] = time_df["timestamp"].dt.floor("1min")
                time_grouped = time_df.groupby("minute").agg(
                    total=("interaction", "count"),
                    bought=("interaction", lambda x: (x == "picked_product").sum()),
                ).reset_index()
                time_grouped["conversion_pct"] = (time_grouped["bought"] / time_grouped["total"] * 100).round(1)
                fig_time = px.line(
                    time_grouped,
                    x="minute",
                    y="conversion_pct",
                    title="Purchase Conversion % Per Minute",
                    labels={"minute": "Time", "conversion_pct": "Conversion %"},
                    markers=True,
                )
                fig_time.update_layout(yaxis_range=[0, 100])
                st.plotly_chart(fig_time, use_container_width=True)

        # ── Recent interactions log ──
        st.subheader("Recent Interactions")
        display_cols = ["timestamp", "customer_id", "shelf_id", "dwell_time_seconds", "status"]
        available_cols = [c for c in display_cols if c in filtered.columns]
        recent = filtered.sort_values("timestamp", ascending=False).head(50)
        st.dataframe(recent[available_cols], use_container_width=True, hide_index=True)

        # ── Taken-item images gallery ──
        has_images = "item_image_path" in filtered.columns
        if has_images:
            taken = filtered[filtered["item_image_path"].notna() & (filtered["item_image_path"] != "")]
            taken = taken.sort_values("timestamp", ascending=False).head(12)
            if not taken.empty:
                st.subheader("📸 Taken Items (Disappearance Snapshots)")
                cols = st.columns(min(4, len(taken)))
                for idx, (_, row) in enumerate(taken.iterrows()):
                    col = cols[idx % len(cols)]
                    img_path = Path(row["item_image_path"])
                    if img_path.exists():
                        col.image(str(img_path), caption=f'{row.get("shelf_id", "")} — {row.get("timestamp", "")}', use_container_width=True)
                    else:
                        col.warning(f"Image not found: {img_path.name}")
    else:
        st.info("No customer interaction data yet.")
else:
    st.info("No customer data available.")


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

st.caption("Dashboard auto-refreshes every 30 seconds (data cached with TTL=30s).")
st.button("Refresh now", on_click=st.cache_data.clear)
