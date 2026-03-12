import logging
from typing import List, Dict, Any

import numpy as np
from ultralytics import YOLO

from shared.config import MODEL_PATHS, PRODUCT_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Model loading (lazy singletons)
# ──────────────────────────────────────────────

_trained_model: YOLO | None = None
_general_model: YOLO | None = None

# COCO classes that look like shelf items (objects you'd find in a store)
_SHELF_ITEM_CLASSES = {
    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
    44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
    49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 73: "book", 74: "clock", 75: "vase",
    76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush",
}

# Lower confidence for general model since it's a fallback
_GENERAL_CONFIDENCE_THRESHOLD = 0.3


def _load_trained_model() -> YOLO:
    """Load the custom-trained inventory YOLO model (once)."""
    global _trained_model
    if _trained_model is None:
        model_path = MODEL_PATHS["inventory"]
        logger.info("Loading trained product model from: %s", model_path)
        _trained_model = YOLO(model_path)
        logger.info("Trained product model loaded successfully.")
    return _trained_model


def _load_general_model() -> YOLO:
    """Load the pretrained YOLOv8 model for general object detection (once)."""
    global _general_model
    if _general_model is None:
        model_path = MODEL_PATHS["person"]  # yolov8n.pt — pretrained on 80 COCO classes
        logger.info("Loading general object model from: %s", model_path)
        _general_model = YOLO(model_path)
        logger.info("General object model loaded successfully.")
    return _general_model


# ──────────────────────────────────────────────
# Detection function
# ──────────────────────────────────────────────

def detect_products(frame: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect products in a single frame using a dual-model approach:

    1. Run the **trained model** first — identifies the 14 known product classes.
    2. Run the **general model** (pretrained YOLOv8) — catches any other
       shelf item that the trained model missed.

    Detections from the general model that overlap with trained-model
    detections are suppressed to avoid duplicates.

    Args:
        frame: A BGR image as a numpy array.

    Returns:
        List of dicts, each containing:
            - product_name (str) — specific name or "unknown_item"
            - bbox (list[float]): [x1, y1, x2, y2]
            - confidence (float)
    """
    detections: List[Dict[str, Any]] = []

    # ── Pass 1: Trained model (known products) ──
    trained_model = _load_trained_model()
    trained_results = trained_model(frame, verbose=False)

    trained_bboxes: List[List[float]] = []

    for result in trained_results:
        boxes = result.boxes
        for box in boxes:
            confidence = float(box.conf[0])
            if confidence < PRODUCT_CONFIDENCE_THRESHOLD:
                continue

            class_id = int(box.cls[0])
            product_name = trained_model.names[class_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = [x1, y1, x2, y2]

            detections.append({
                "product_name": product_name,
                "bbox": bbox,
                "confidence": confidence,
            })
            trained_bboxes.append(bbox)

    # ── Pass 2: General model (catch unknown items) ──
    general_model = _load_general_model()
    general_results = general_model(frame, verbose=False)

    for result in general_results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            # Skip persons (class 0) and non-shelf-item classes
            if class_id == 0:
                continue

            confidence = float(box.conf[0])
            if confidence < _GENERAL_CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = [x1, y1, x2, y2]

            # Skip if this detection overlaps significantly with a trained-model detection
            if _overlaps_any(bbox, trained_bboxes, iou_threshold=0.3):
                continue

            # Use COCO class name if it's a known shelf item, otherwise "unknown_item"
            if class_id in _SHELF_ITEM_CLASSES:
                product_name = _SHELF_ITEM_CLASSES[class_id]
            else:
                product_name = "unknown_item"

            detections.append({
                "product_name": product_name,
                "bbox": bbox,
                "confidence": confidence,
            })

    logger.info("Detected %d items in frame (%d known, %d general).",
                len(detections), len(trained_bboxes),
                len(detections) - len(trained_bboxes))
    return detections


def _iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute Intersection over Union between two [x1,y1,x2,y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _overlaps_any(bbox: List[float], existing: List[List[float]],
                  iou_threshold: float = 0.3) -> bool:
    """Return True if bbox overlaps any existing bbox above the threshold."""
    for ex in existing:
        if _iou(bbox, ex) > iou_threshold:
            return True
    return False
