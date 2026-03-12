"""
Product disappearance tracker.

Compares product bounding boxes across consecutive detection frames to
find items that were present before but are now gone.  When an item
disappears, the system crops that region from the **previous** frame
and saves the image — giving a clear photo of the taken product.

Flow:
    Frame N:   Detect products → list of bboxes [A, B, C, D, E]
    Frame N+k: Detect products → list of bboxes [A, B, D, E]
               → "C" disappeared → crop C's region from frame N
               → save crop to  taken_items/<timestamp>.jpg
               → return event with shelf_id, bbox, and image path
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Directory where cropped images of taken items are saved
TAKEN_ITEMS_DIR = Path("taken_items")
TAKEN_ITEMS_DIR.mkdir(exist_ok=True)

# IoU threshold to consider two boxes as "the same product"
_MATCH_IOU_THRESHOLD = 0.3

# A product must be missing for this many detection cycles to count
_MIN_MISSING_CYCLES = 2

# Minimum crop size (px) — crops below this are upscaled
_MIN_CROP_PX = 128

# Padding factor around the bbox (fraction of bbox size) for wider context
_PAD_FACTOR = 0.35

# State ───────────────────────────────────────
_prev_detections: List[Dict[str, Any]] = []
_prev_frame: Optional[np.ndarray] = None

# Tracks boxes that went missing — bbox_key -> {"bbox", "missing_count", "frame"}
_missing_candidates: Dict[str, Dict] = {}


def _iou(a: List[float], b: List[float]) -> float:
    """Intersection over Union between two [x1,y1,x2,y2] boxes."""
    xa = max(a[0], b[0])
    ya = max(a[1], b[1])
    xb = min(a[2], b[2])
    yb = min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter) if (area_a + area_b - inter) > 0 else 0.0


def _bbox_key(bbox: List[float]) -> str:
    """Stable string key for a bbox (rounded to avoid float noise)."""
    return f"{bbox[0]:.0f}_{bbox[1]:.0f}_{bbox[2]:.0f}_{bbox[3]:.0f}"


def _match_detections(
    prev: List[Dict], curr: List[Dict]
) -> Tuple[List[Dict], List[Dict]]:
    """
    Match previous detections to current ones by IoU.

    Returns:
        (unmatched_prev, unmatched_curr)
        - unmatched_prev: items in prev that have no match in curr (disappeared)
        - unmatched_curr: items in curr that have no match in prev (appeared)
    """
    if not prev:
        return [], curr
    if not curr:
        return prev, []

    used_curr = set()
    matched_prev = set()

    for i, p in enumerate(prev):
        best_j = -1
        best_iou = _MATCH_IOU_THRESHOLD
        for j, c in enumerate(curr):
            if j in used_curr:
                continue
            score = _iou(p["bbox"], c["bbox"])
            if score > best_iou:
                best_iou = score
                best_j = j
        if best_j >= 0:
            used_curr.add(best_j)
            matched_prev.add(i)

    unmatched_prev = [prev[i] for i in range(len(prev)) if i not in matched_prev]
    unmatched_curr = [curr[j] for j in range(len(curr)) if j not in used_curr]
    return unmatched_prev, unmatched_curr


def _enhance_crop(crop: np.ndarray) -> np.ndarray:
    """Upscale, denoise, and sharpen a small crop so it's identifiable."""
    h, w = crop.shape[:2]

    # 1. Upscale small crops so the shopkeeper can actually see the item
    short_side = min(h, w)
    if short_side < _MIN_CROP_PX:
        scale = max(2, _MIN_CROP_PX / short_side)
        # Cap at 4× to avoid blowing up memory
        scale = min(scale, 4.0)
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_LANCZOS4)

    # 2. Gentle denoise (removes JPEG / sensor noise without smearing edges)
    crop = cv2.fastNlMeansDenoisingColored(crop, None, h=6, hForColorComponents=6,
                                            templateWindowSize=7, searchWindowSize=21)

    # 3. Unsharp-mask sharpening: blur → weighted subtract → crisp edges
    blurred = cv2.GaussianBlur(crop, (0, 0), sigmaX=2)
    crop = cv2.addWeighted(crop, 1.5, blurred, -0.5, 0)

    return crop


def _save_crop(frame: np.ndarray, bbox: List[float], shelf_id: str) -> str:
    """Crop the bbox region (with context padding), enhance, and save as PNG."""
    h, w = frame.shape[:2]

    # Wider crop — add padding proportional to bbox size for surrounding context
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    pad_x = int(bw * _PAD_FACTOR)
    pad_y = int(bh * _PAD_FACTOR)

    x1 = max(0, int(bbox[0]) - pad_x)
    y1 = max(0, int(bbox[1]) - pad_y)
    x2 = min(w, int(bbox[2]) + pad_x)
    y2 = min(h, int(bbox[3]) + pad_y)
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return ""

    crop = _enhance_crop(crop)

    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{shelf_id}_{x1}_{y1}.png"
    filepath = str(TAKEN_ITEMS_DIR / filename)
    # PNG = lossless — no extra JPEG compression artefacts
    cv2.imwrite(filepath, crop)
    logger.info("Saved taken-item crop: %s  (%dx%d)", filepath, crop.shape[1], crop.shape[0])
    return filepath


def check_disappeared(
    current_detections: List[Dict[str, Any]],
    current_frame: np.ndarray,
    shelf_regions: List[Dict] = None,
) -> List[Dict[str, Any]]:
    """
    Compare current product detections against the previous frame.

    Returns a list of "taken item" events, each containing:
        - bbox: [x1, y1, x2, y2] of the missing product
        - shelf_id: which shelf it was on
        - image_path: path to the saved crop image
        - timestamp: when it was detected as missing
    """
    global _prev_detections, _prev_frame

    if shelf_regions is None:
        shelf_regions = []

    events: List[Dict[str, Any]] = []

    if _prev_frame is not None and _prev_detections:
        disappeared, _appeared = _match_detections(_prev_detections, current_detections)

        # Track disappearances over multiple cycles for robustness
        current_keys = {_bbox_key(d["bbox"]) for d in current_detections}

        for det in disappeared:
            key = _bbox_key(det["bbox"])
            if key in _missing_candidates:
                _missing_candidates[key]["missing_count"] += 1
            else:
                _missing_candidates[key] = {
                    "bbox": det["bbox"],
                    "product_name": det.get("product_name", "product"),
                    "missing_count": 1,
                    "frame": _prev_frame,
                }

        # Check which candidates are confirmed missing
        confirmed_keys = []
        for key, info in list(_missing_candidates.items()):
            if key in current_keys:
                # Item reappeared — false alarm
                del _missing_candidates[key]
                continue

            if info["missing_count"] >= _MIN_MISSING_CYCLES:
                bbox = info["bbox"]
                shelf_id = _get_shelf_for_bbox(bbox, shelf_regions)
                image_path = _save_crop(info["frame"], bbox, shelf_id)

                if image_path:
                    events.append({
                        "bbox": bbox,
                        "shelf_id": shelf_id,
                        "image_path": image_path,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    })
                confirmed_keys.append(key)

        for key in confirmed_keys:
            _missing_candidates.pop(key, None)

    # Update state for next frame
    _prev_detections = current_detections
    _prev_frame = current_frame.copy()

    if events:
        logger.info("Detected %d taken item(s) this cycle.", len(events))

    return events


def _get_shelf_for_bbox(
    bbox: List[float], shelf_regions: List[Dict]
) -> str:
    """Return the shelf_id that contains the centre of the bbox."""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    for region in shelf_regions:
        if (region.get("x_min", 0) <= cx <= region.get("x_max", 0) and
                region.get("y_min", 0) <= cy <= region.get("y_max", 0)):
            return region.get("shelf_id") or region.get("name", "shelf_unknown")
    return "shelf_unknown"


def reset():
    """Clear all state (for tests)."""
    global _prev_detections, _prev_frame
    _prev_detections = []
    _prev_frame = None
    _missing_candidates.clear()
