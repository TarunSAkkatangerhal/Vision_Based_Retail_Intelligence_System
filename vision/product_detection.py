import logging
from typing import List, Dict, Any

import numpy as np
from ultralytics import YOLO

from shared.config import MODEL_PATHS, PRODUCT_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Model loading (lazy singleton)
# ──────────────────────────────────────────────

_model: YOLO | None = None


def _load_model() -> YOLO:
    """Load the trained inventory YOLO model (once)."""
    global _model
    if _model is None:
        model_path = MODEL_PATHS["inventory"]
        logger.info("Loading product detection model from: %s", model_path)
        _model = YOLO(model_path)
        logger.info("Product detection model loaded successfully.")
    return _model


# ──────────────────────────────────────────────
# Detection function
# ──────────────────────────────────────────────

def detect_products(frame: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect products in a single frame.

    Args:
        frame: A BGR image as a numpy array.

    Returns:
        List of dicts, each containing:
            - product_name (str)
            - bbox (list[float]): [x1, y1, x2, y2]
            - confidence (float)
    """
    model = _load_model()
    results = model(frame, verbose=False)

    detections: List[Dict[str, Any]] = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = float(box.conf[0])
            if confidence < PRODUCT_CONFIDENCE_THRESHOLD:
                continue

            class_id = int(box.cls[0])
            product_name = model.names[class_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "product_name": product_name,
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
            })

    logger.info("Detected %d products in frame.", len(detections))
    return detections
