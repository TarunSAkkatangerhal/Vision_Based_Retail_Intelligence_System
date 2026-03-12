import logging
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
from ultralytics import YOLO

from shared.config import MODEL_PATHS, PRODUCT_CONFIDENCE_THRESHOLD, PRODUCT_CLASSES

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Model loading (lazy singleton)
# ──────────────────────────────────────────────

_product_model: YOLO | None = None
_model_is_trained: bool = False   # True → custom trained model, False → YOLOWorld

# Path to the trained model produced by dataset/train_sku110k.py
_TRAINED_MODEL_PATH = Path("models/inventory_yolo.pt")

# Inference image size — larger helps detect small shelf items
_INFER_SIZE = 1280

# If a frame is smaller than this width, upscale it for better detection
_MIN_WIDTH_FOR_DETECTION = 640
_MAX_UPSCALE = 3.0  # allow up to 3x for very low-res feeds

# SAHI-style tiled detection settings
_TILE_ENABLED = True          # Enable tiled detection for small products
_TILE_OVERLAP_RATIO = 0.35    # 35% overlap between tiles (more overlap = fewer misses)
_NMS_IOU_THRESHOLD = 0.45     # IoU threshold for merging duplicate detections

# Use half-precision (FP16) if a CUDA GPU is available
_USE_HALF = False
try:
    import torch
    _USE_HALF = torch.cuda.is_available()
except ImportError:
    pass


def _load_model() -> YOLO:
    """
    Load the best available product detection model (once).

    Priority:
      1. Custom trained model (models/inventory_yolo.pt) — trained on SKU-110K,
         single-class "product" detector, highly accurate for dense shelves.
      2. YOLOWorld open-vocabulary model — zero-shot fallback when no trained
         model exists yet.
    """
    global _product_model, _model_is_trained

    if _product_model is not None:
        return _product_model

    if _TRAINED_MODEL_PATH.exists():
        logger.info("Loading TRAINED product model from: %s", _TRAINED_MODEL_PATH)
        _product_model = YOLO(str(_TRAINED_MODEL_PATH))
        _model_is_trained = True
        logger.info("Trained model ready — single-class dense product detector.")
    else:
        model_path = MODEL_PATHS["inventory"]
        logger.info("No trained model found at %s — using YOLOWorld fallback: %s",
                     _TRAINED_MODEL_PATH, model_path)
        _product_model = YOLO(model_path)
        _product_model.set_classes(PRODUCT_CLASSES)
        _model_is_trained = False
        logger.info("YOLOWorld ready — detecting classes: %s", PRODUCT_CLASSES)

    return _product_model


# ──────────────────────────────────────────────
# Detection function
# ──────────────────────────────────────────────

def detect_products(frame: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect products in a single frame using YOLOWorld open-vocabulary detection.

    Uses a two-pass approach for reliability:
      1. Full-frame detection at high resolution (catches large/medium items).
      2. SAHI-style tiled detection (catches small items on distant shelves).

    Duplicate detections from overlapping tiles are merged via NMS.

    Args:
        frame: A BGR image as a numpy array.

    Returns:
        List of dicts, each containing:
            - product_name (str)
            - bbox (list[float]): [x1, y1, x2, y2]
            - confidence (float)
    """
    model = _load_model()

    # Trained model is much more accurate — use a higher threshold to reduce noise
    conf_threshold = 0.35 if _model_is_trained else PRODUCT_CONFIDENCE_THRESHOLD

    # Upscale low-resolution frames so small products on shelves are visible
    h, w = frame.shape[:2]
    scale = 1.0
    detect_frame = frame
    if w < _MIN_WIDTH_FOR_DETECTION:
        scale = min(_MIN_WIDTH_FOR_DETECTION / w, _MAX_UPSCALE)
        detect_frame = cv2.resize(
            frame, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )
        logger.debug("Upscaled frame %.1fx for product detection.", scale)

    # Enhance contrast — helps with glass-door refrigerators / dim shelves
    # Only apply mild sharpening; avoid CLAHE as it can distort colours
    # and hurt open-vocabulary text-image matching.
    detect_frame = _sharpen(detect_frame)

    dh, dw = detect_frame.shape[:2]

    all_detections: List[Dict[str, Any]] = []

    # ── Pass 1: Full-frame detection ──
    results = model(detect_frame, verbose=False, imgsz=_INFER_SIZE, half=_USE_HALF)
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence < conf_threshold:
                continue
            class_id = int(box.cls[0])
            product_name = model.names[class_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            all_detections.append({
                "product_name": product_name,
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
            })

    # ── Pass 2: Tiled detection (SAHI approach) ──
    if _TILE_ENABLED:
        # Use a 3x3-ish grid: tiles are ~1/3 of each dimension
        tile_h = max(dh // 3, 320)
        tile_w = max(dw // 2, 320)
        step_h = int(tile_h * (1 - _TILE_OVERLAP_RATIO))
        step_w = int(tile_w * (1 - _TILE_OVERLAP_RATIO))

        for y in range(0, max(1, dh - tile_h + 1), step_h):
            y_end = min(y + tile_h, dh)
            for x in range(0, max(1, dw - tile_w + 1), step_w):
                x_end = min(x + tile_w, dw)
                tile = detect_frame[y:y_end, x:x_end]
                if tile.size == 0:
                    continue

                tile_results = model(tile, verbose=False, imgsz=640, half=_USE_HALF)
                for result in tile_results:
                    for box in result.boxes:
                        confidence = float(box.conf[0])
                        if confidence < conf_threshold:
                            continue
                        class_id = int(box.cls[0])
                        product_name = model.names[class_id]
                        tx1, ty1, tx2, ty2 = box.xyxy[0].tolist()
                        # Map tile coords back to full (upscaled) frame
                        all_detections.append({
                            "product_name": product_name,
                            "bbox": [tx1 + x, ty1 + y, tx2 + x, ty2 + y],
                            "confidence": confidence,
                        })

    # ── NMS to merge duplicates from overlapping tiles ──
    if len(all_detections) > 1:
        all_detections = _nms_merge(all_detections, _NMS_IOU_THRESHOLD)

    # ── Scale coordinates back to original frame ──
    if scale != 1.0:
        for det in all_detections:
            det["bbox"] = [c / scale for c in det["bbox"]]

    logger.info("Detected %d products in frame.", len(all_detections))
    return all_detections


def _sharpen(image: np.ndarray) -> np.ndarray:
    """Mild sharpening to bring out shelf product edges without distorting colours."""
    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3, -0.5],
                       [0, -0.5, 0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def _enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement to help detect items behind glass / dim shelves."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _nms_merge(
    detections: List[Dict[str, Any]],
    iou_threshold: float,
) -> List[Dict[str, Any]]:
    """
    Non-maximum suppression to remove duplicate detections from tiled passes.

    Keeps the higher-confidence detection when two boxes overlap significantly.
    """
    if not detections:
        return detections

    # Sort by confidence descending
    detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    keep = []

    while detections:
        best = detections.pop(0)
        keep.append(best)
        remaining = []
        for det in detections:
            if _iou(best["bbox"], det["bbox"]) < iou_threshold:
                remaining.append(det)
        detections = remaining

    return keep


def _iou(box_a: list, box_b: list) -> float:
    """Compute Intersection over Union between two [x1,y1,x2,y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
