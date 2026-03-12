import logging
from typing import Dict, Any, List, Tuple

import numpy as np

from shared.config import (
    CAMERA_ID,
    MOCK_MODE,
    MODEL_PATHS,
    PERSON_CONFIDENCE_THRESHOLD,
    DWELL_TIME_THRESHOLD,
)
from shared.schemas import CustomerBehaviorData, CustomerItem
from shared.utils import get_current_timestamp
from vision.dwell_time import calculate_dwell_time
from vision.interaction import detect_interaction

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Shelf regions (pixel-space) — must match inventory.py mapping
# ──────────────────────────────────────────────
SHELF_REGIONS: List[Dict] = [
    {"shelf_id": "shelf_A", "x_min": 0, "x_max": 213, "y_min": 0, "y_max": 480},
    {"shelf_id": "shelf_B", "x_min": 213, "x_max": 426, "y_min": 0, "y_max": 480},
    {"shelf_id": "shelf_C", "x_min": 426, "x_max": 640, "y_min": 0, "y_max": 480},
]

# ──────────────────────────────────────────────
# Lazy-loaded singletons
# ──────────────────────────────────────────────
_yolo_model = None
_tracker = None

# Per-track position history: track_id -> list[(cx, cy)]
_track_positions: Dict[str, List[Tuple[float, float]]] = {}


def _load_person_model():
    """Load the pretrained YOLOv8 person detection model (once)."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO

        model_path = MODEL_PATHS["person"]
        logger.info("Loading person detection model: %s", model_path)
        _yolo_model = YOLO(model_path)
        logger.info("Person detection model loaded successfully.")
    return _yolo_model


def _get_tracker():
    """Initialise a DeepSort tracker (once)."""
    global _tracker
    if _tracker is None:
        from deep_sort_realtime.deepsort_tracker import DeepSort
        _tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
        logger.info("DeepSort tracker initialised.")
    return _tracker


def _get_shelf_for_position(cx: float, cy: float) -> str:
    """Return the shelf_id that contains the given centre point."""
    for region in SHELF_REGIONS:
        if region["x_min"] <= cx <= region["x_max"] and region["y_min"] <= cy <= region["y_max"]:
            return region["shelf_id"]
    return "shelf_unknown"


# ──────────────────────────────────────────────
# Mock data (MOCK_MODE)
# ──────────────────────────────────────────────

def _mock_customers() -> dict:
    """Return hardcoded sample data matching the Customer Behavior Schema."""
    data = CustomerBehaviorData(
        schema_version="1.0",
        timestamp=get_current_timestamp(),
        camera_id=CAMERA_ID,
        customers=[
            CustomerItem(
                customer_id="cust_001",
                shelf_id="shelf_A",
                dwell_time_seconds=8,
                interaction="picked_product",
            ),
            CustomerItem(
                customer_id="cust_002",
                shelf_id="shelf_B",
                dwell_time_seconds=3,
                interaction="none",
            ),
            CustomerItem(
                customer_id="cust_003",
                shelf_id="shelf_C",
                dwell_time_seconds=12,
                interaction="replaced_product",
            ),
        ],
    )
    logger.info("MOCK_MODE: returning hardcoded customer data.")
    return data.model_dump()


# ──────────────────────────────────────────────
# Main detection entry-point
# ──────────────────────────────────────────────

def detect_customers(frame: np.ndarray) -> Dict[str, Any]:
    """
    Detect and track customers in a single frame.

    When MOCK_MODE is True the function returns hardcoded sample data
    without loading any model or using GPU.

    Returns:
        dict matching the Customer Behavior Schema.
    """
    if MOCK_MODE:
        return _mock_customers()

    model = _load_person_model()
    tracker = _get_tracker()

    # ── YOLOv8 person detection ──
    results = model(frame, verbose=False)

    # Build a list of detections in the format DeepSort expects:
    # Each detection is ([left, top, w, h], confidence, class_name)
    raw_detections: List[Tuple[List[float], float, str]] = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            # COCO class 0 == "person"
            if class_id != 0:
                continue
            confidence = float(box.conf[0])
            if confidence < PERSON_CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1
            raw_detections.append(([x1, y1, w, h], confidence, "person"))

    # ── DeepSort tracking ──
    tracks = tracker.update_tracks(raw_detections, frame=frame)

    customers: List[CustomerItem] = []

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = str(track.track_id)
        ltrb = track.to_ltrb()  # [left, top, right, bottom]
        cx = (ltrb[0] + ltrb[2]) / 2
        cy = (ltrb[1] + ltrb[3]) / 2
        position = (cx, cy)

        # Accumulate position history for interaction detection
        if track_id not in _track_positions:
            _track_positions[track_id] = []
        _track_positions[track_id].append(position)

        # Keep history bounded
        _track_positions[track_id] = _track_positions[track_id][-60:]

        # Determine which shelf this person is near
        shelf_id = _get_shelf_for_position(cx, cy)

        # Dwell time
        dwell = calculate_dwell_time(track_id, position, SHELF_REGIONS)

        # Interaction
        interaction = detect_interaction(track_id, _track_positions[track_id])

        customers.append(
            CustomerItem(
                customer_id=f"cust_{track_id}",
                shelf_id=shelf_id,
                dwell_time_seconds=dwell,
                interaction=interaction,
            )
        )

    behaviour = CustomerBehaviorData(
        schema_version="1.0",
        timestamp=get_current_timestamp(),
        camera_id=CAMERA_ID,
        customers=customers,
    )

    # Validate via Pydantic and return dict
    validated = CustomerBehaviorData.model_validate(behaviour.model_dump())
    logger.info("Detected %d customers in frame.", len(validated.customers))
    return validated.model_dump()
