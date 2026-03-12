"""
Auto Shelf Detection — Cluster detected products into shelf regions.

Instead of manually calibrating shelf boundaries, this module groups
product detections that are spatially close (bunched together) into
automatic shelf clusters using DBSCAN.

Flow:
  1. Collect product bounding-box centres from detection results.
  2. Run DBSCAN clustering on the 2-D centres.
  3. For each cluster, compute the bounding hull → that's one shelf region.
  4. Merge with any manually-configured shelves from shelf_config.json.
  5. Expose the regions in the same format the rest of the pipeline expects.

The auto-detection runs on the first N "warm-up" frames and then
stabilises.  It can also be re-triggered if the product layout changes
significantly (e.g. restocking).
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Tunables ──
_CLUSTER_EPS = 80           # max pixel distance between items in one shelf group
_CLUSTER_MIN_SAMPLES = 2    # min products to form a shelf cluster
_PADDING = 15               # pixels of padding around each auto-detected shelf
_WARMUP_FRAMES = 5          # accumulate detections for this many frames before clustering
_RECLUSTER_INTERVAL = 30.0  # re-cluster every N seconds to adapt

# ── State ──
_warmup_detections: List[List[float]] = []   # accumulated bbox centres during warmup
_warmup_count = 0
_auto_shelves: List[Dict] = []
_last_cluster_time: float = 0.0
_is_stable = False


def _dbscan_cluster(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """
    Simple DBSCAN implementation (no sklearn dependency).

    Returns an array of cluster labels (-1 = noise).
    """
    n = len(points)
    labels = np.full(n, -1, dtype=int)
    cluster_id = 0

    visited = np.zeros(n, dtype=bool)

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        # Region query — find neighbours within eps
        dists = np.linalg.norm(points - points[i], axis=1)
        neighbours = np.where(dists <= eps)[0]

        if len(neighbours) < min_samples:
            continue  # noise point

        labels[i] = cluster_id
        seed_set = list(neighbours)
        j = 0
        while j < len(seed_set):
            q = seed_set[j]
            if not visited[q]:
                visited[q] = True
                q_dists = np.linalg.norm(points - points[q], axis=1)
                q_neighbours = np.where(q_dists <= eps)[0]
                if len(q_neighbours) >= min_samples:
                    seed_set.extend(q_neighbours.tolist())
            if labels[q] == -1:
                labels[q] = cluster_id
            j += 1

        cluster_id += 1

    return labels


def _build_shelf_regions(
    points: np.ndarray,
    labels: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> List[Dict]:
    """Convert DBSCAN clusters into shelf region dicts."""
    shelves = []
    unique_labels = set(labels)
    unique_labels.discard(-1)

    for cluster_id in sorted(unique_labels):
        mask = labels == cluster_id
        cluster_points = points[mask]

        x_min = max(0, int(cluster_points[:, 0].min()) - _PADDING)
        y_min = max(0, int(cluster_points[:, 1].min()) - _PADDING)
        x_max = min(frame_w, int(cluster_points[:, 0].max()) + _PADDING)
        y_max = min(frame_h, int(cluster_points[:, 1].max()) + _PADDING)

        shelf_name = f"auto_shelf_{chr(65 + cluster_id)}"  # A, B, C, ...
        shelves.append({
            "shelf_id": shelf_name,
            "name": shelf_name,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "auto_detected": True,
            "product_count": int(mask.sum()),
        })

    logger.info("Auto-detected %d shelf region(s) from %d product detections.",
                len(shelves), len(points))
    for s in shelves:
        logger.info("  %s: (%d,%d)-(%d,%d) [%d products]",
                     s["shelf_id"], s["x_min"], s["y_min"],
                     s["x_max"], s["y_max"], s["product_count"])
    return shelves


def update_auto_shelves(
    product_detections: List[Dict],
    frame_width: int,
    frame_height: int,
    force: bool = False,
) -> List[Dict]:
    """
    Feed product detections and return auto-detected shelf regions.

    During warmup (first N frames), centres are accumulated. After warmup
    or when `force=True`, DBSCAN runs and shelf regions are produced.

    Returns:
        List of shelf region dicts with keys:
        shelf_id, name, x_min, y_min, x_max, y_max, auto_detected
    """
    global _warmup_count, _is_stable, _auto_shelves, _last_cluster_time

    now = time.time()

    # Extract centres from product bounding boxes
    centres = []
    for det in product_detections:
        bbox = det["bbox"]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        centres.append([cx, cy])

    if not centres:
        return _auto_shelves  # nothing to work with

    _warmup_detections.extend(centres)
    _warmup_count += 1

    # Decide whether to (re)cluster
    should_cluster = force
    if not _is_stable and _warmup_count >= _WARMUP_FRAMES:
        should_cluster = True
    elif _is_stable and (now - _last_cluster_time) > _RECLUSTER_INTERVAL:
        should_cluster = True

    if not should_cluster:
        return _auto_shelves

    # Run DBSCAN on all accumulated points
    all_points = np.array(_warmup_detections, dtype=np.float32)

    if len(all_points) < _CLUSTER_MIN_SAMPLES:
        return _auto_shelves

    labels = _dbscan_cluster(all_points, _CLUSTER_EPS, _CLUSTER_MIN_SAMPLES)
    _auto_shelves = _build_shelf_regions(all_points, labels, frame_width, frame_height)
    _last_cluster_time = now
    _is_stable = True

    # Keep only recent detections to allow drift over time
    max_keep = _WARMUP_FRAMES * 50  # roughly 50 items per frame max            
    if len(_warmup_detections) > max_keep:
        _warmup_detections[:] = _warmup_detections[-max_keep:]

    return _auto_shelves


def get_auto_shelves() -> List[Dict]:
    """Return the current auto-detected shelf regions."""
    return list(_auto_shelves)


def is_stable() -> bool:
    """Return True if auto-detection has completed warm-up and produced regions."""
    return _is_stable


def reset() -> None:
    """Clear all state (useful for tests)."""
    global _warmup_count, _is_stable, _auto_shelves, _last_cluster_time
    _warmup_detections.clear()
    _warmup_count = 0
    _auto_shelves = []
    _last_cluster_time = 0.0
    _is_stable = False
