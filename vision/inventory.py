import logging
from collections import defaultdict
from typing import Dict, Any, List

import numpy as np

from shared.config import (
    CAMERA_ID,
    LOW_STOCK_THRESHOLD,
    MOCK_MODE,
)
from shared.schemas import InventoryData, ProductItem
from shared.utils import get_current_timestamp

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Shelf region mapping
# ──────────────────────────────────────────────
# Maps pixel-space x-ranges to logical shelf IDs.
# Adjust these regions to match your camera layout.
SHELF_REGIONS = {
    "shelf_A": (0, 213),
    "shelf_B": (213, 426),
    "shelf_C": (426, 640),
}

# ──────────────────────────────────────────────
# Shelf state tracking (across frames)
# ──────────────────────────────────────────────
# Previous frame's item count per shelf — used to detect changes
_previous_shelf_counts: Dict[str, int] = {}
_shelf_change_log: List[str] = []


def get_shelf_changes() -> List[str]:
    """Return the list of shelf change events detected so far."""
    return list(_shelf_change_log)


def reset_shelf_tracking() -> None:
    """Clear shelf tracking state (useful for tests)."""
    _previous_shelf_counts.clear()
    _shelf_change_log.clear()


def _assign_shelf(bbox: list[float]) -> str:
    """Return the shelf_id for a detection based on bbox centre-x."""
    cx = (bbox[0] + bbox[2]) / 2
    for shelf_id, (x_min, x_max) in SHELF_REGIONS.items():
        if x_min <= cx < x_max:
            return shelf_id
    return "shelf_unknown"


# ──────────────────────────────────────────────
# Mock data (used when MOCK_MODE is True)
# ──────────────────────────────────────────────

def _mock_inventory() -> dict:
    """Return hardcoded sample data matching the Inventory Schema."""
    data = InventoryData(
        schema_version="1.0",
        timestamp=get_current_timestamp(),
        camera_id=CAMERA_ID,
        products=[
            ProductItem(
                product_name="product_a",
                shelf_id="shelf_A",
                count=5,
                status="normal",
            ),
            ProductItem(
                product_name="product_b",
                shelf_id="shelf_B",
                count=2,
                status="low_stock",
            ),
            ProductItem(
                product_name="product_c",
                shelf_id="shelf_C",
                count=0,
                status="empty",
            ),
        ],
    )
    logger.info("MOCK_MODE: returning hardcoded inventory data.")
    return data.model_dump()


# ──────────────────────────────────────────────
# Main detection entry-point
# ──────────────────────────────────────────────

def detect_inventory(frame: np.ndarray) -> Dict[str, Any]:
    """
    Analyse a single frame and return an Inventory Schema dict.

    Uses a dual-model approach:
      - Trained model identifies the 14 known product classes by name.
      - General model catches ANY other item on the shelf as "unknown_item".

    Also tracks shelf state changes across frames and logs events like
    "item taken from shelf_A" or "shelf_B is now empty".

    When MOCK_MODE is True the function returns hardcoded sample data
    without loading any model.
    """
    if MOCK_MODE:
        return _mock_inventory()

    # Import here so the model is only loaded when actually needed
    from vision.product_detection import detect_products

    detections = detect_products(frame)

    # ── Group detections by shelf ──
    shelf_total_counts: Dict[str, int] = defaultdict(int)
    shelf_product_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for det in detections:
        shelf_id = _assign_shelf(det["bbox"])
        shelf_product_counts[shelf_id][det["product_name"]] += 1
        shelf_total_counts[shelf_id] += 1

    # Ensure all known shelves appear in counts (even if 0)
    for shelf_id in SHELF_REGIONS:
        if shelf_id not in shelf_total_counts:
            shelf_total_counts[shelf_id] = 0

    # ── Detect shelf state changes ──
    timestamp = get_current_timestamp()
    for shelf_id in SHELF_REGIONS:
        current_count = shelf_total_counts[shelf_id]
        previous_count = _previous_shelf_counts.get(shelf_id, -1)

        if previous_count == -1:
            # First frame — just record baseline
            pass
        elif current_count < previous_count:
            diff = previous_count - current_count
            event = (f"[{timestamp}] {diff} item(s) TAKEN from {shelf_id} "
                     f"(was {previous_count}, now {current_count})")
            _shelf_change_log.append(event)
            logger.warning(event)
            if current_count == 0:
                event_empty = f"[{timestamp}] {shelf_id} is now EMPTY!"
                _shelf_change_log.append(event_empty)
                logger.warning(event_empty)
        elif current_count > previous_count and previous_count >= 0:
            diff = current_count - previous_count
            event = (f"[{timestamp}] {diff} item(s) PLACED on {shelf_id} "
                     f"(was {previous_count}, now {current_count})")
            _shelf_change_log.append(event)
            logger.info(event)

        _previous_shelf_counts[shelf_id] = current_count

    # ── Build product list ──
    products: list[ProductItem] = []
    for shelf_id, product_map in shelf_product_counts.items():
        for product_name, count in product_map.items():
            if count == 0:
                status = "empty"
            elif count <= LOW_STOCK_THRESHOLD:
                status = "low_stock"
            else:
                status = "normal"

            products.append(
                ProductItem(
                    product_name=product_name,
                    shelf_id=shelf_id,
                    count=count,
                    status=status,
                )
            )

    # Report shelves with zero items
    if not products:
        for shelf_id in SHELF_REGIONS:
            products.append(
                ProductItem(
                    product_name="no_items",
                    shelf_id=shelf_id,
                    count=0,
                    status="empty",
                )
            )
    else:
        # Also add empty entries for shelves that had no detections
        shelves_with_products = {p.shelf_id for p in products}
        for shelf_id in SHELF_REGIONS:
            if shelf_id not in shelves_with_products:
                products.append(
                    ProductItem(
                        product_name="no_items",
                        shelf_id=shelf_id,
                        count=0,
                        status="empty",
                    )
                )

    inventory = InventoryData(
        schema_version="1.0",
        timestamp=timestamp,
        camera_id=CAMERA_ID,
        products=products,
    )

    # Validate via Pydantic and return dict
    validated = InventoryData.model_validate(inventory.model_dump())
    total_items = sum(shelf_total_counts.values())
    logger.info("Inventory: %d total items across %d shelves, %d product entries.",
                total_items, len(SHELF_REGIONS), len(validated.products))
    return validated.model_dump()
