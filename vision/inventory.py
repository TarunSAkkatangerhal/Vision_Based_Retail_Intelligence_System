import logging
from collections import defaultdict
from typing import Dict, Any

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

    When MOCK_MODE is True the function returns hardcoded sample data
    without loading any model.
    """
    if MOCK_MODE:
        return _mock_inventory()

    # Import here so the model is only loaded when actually needed
    from vision.product_detection import detect_products

    detections = detect_products(frame)

    # Group detections by (shelf_id, product_name) and count
    shelf_product_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for det in detections:
        shelf_id = _assign_shelf(det["bbox"])
        shelf_product_counts[shelf_id][det["product_name"]] += 1

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

    # If no products detected at all, report every shelf as empty
    if not products:
        for shelf_id in SHELF_REGIONS:
            products.append(
                ProductItem(
                    product_name="unknown",
                    shelf_id=shelf_id,
                    count=0,
                    status="empty",
                )
            )

    inventory = InventoryData(
        schema_version="1.0",
        timestamp=get_current_timestamp(),
        camera_id=CAMERA_ID,
        products=products,
    )

    # Validate via Pydantic and return dict
    validated = InventoryData.model_validate(inventory.model_dump())
    logger.info("Inventory detected: %d product entries.", len(validated.products))
    return validated.model_dump()
